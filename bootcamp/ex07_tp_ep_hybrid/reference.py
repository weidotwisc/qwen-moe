"""Reference implementation for Ex07 — TP-4 × DP-2 × EP-8 hybrid block.

The general case (3) from the design discussion:
- Two TP groups of 4 ranks each: {0,1,2,3} and {4,5,6,7}.
- Each TP group processes a distinct half of the batch (DP-2 outer).
- One EP group over all 8 ranks; MoE dispatch crosses TP-group boundaries.
- Attention runs TP-4 within each TP group.
- MoE stripes tokens within TP group, dispatches over EP group, all_gathers
  within TP group.

Sub-group composition: rank 0 belongs to `tp_group_a` AND `ep_group`.
The paper's composition theorem proves this cross-cutting sub-group
interleaving is deadlock-free under fixed-schedule + explicit-group
discipline.

Contract:
    Input:  x on rank r = [B, T, H] — this rank's TP group's half of
            the batch, replicated across ranks in the same TP group.
    Output: y on rank r = [B, T, H] — same shape, replicated within
            TP group. The block-level invariant "replicated within
            TP group" holds on both sides.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex00_dist_primer.solution import all_to_all_variable
from bootcamp.ex04_gqa_tp.reference import TPGQA
from bootcamp.ref.block import RMSNorm
from bootcamp.ref.mlp import RefSwiGLU_MLP


class HybridMoE(nn.Module):
    """TP-striped, EP-dispatched MoE.

    - Input arrives replicated within `tp_group` (post-attention TP AR).
    - Stripe within tp_group so each rank has a distinct 1/tp_size share.
    - Dispatch across `ep_group` (may cross TP-group boundaries).
    - Local expert compute.
    - Combine across ep_group.
    - All_gather within tp_group to restore the replicated-within-TP contract.

    Args:
        hidden, intermediate, num_experts, top_k, norm_topk_prob — MoE dims.
        tp_size, tp_rank, tp_group — TP within-group parameters.
        ep_size, ep_rank, ep_group — EP world-spanning parameters.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        ep_group: dist.ProcessGroup | None,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        assert num_experts % ep_size == 0
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        self.norm_topk_prob = norm_topk_prob
        self.experts_per_rank = num_experts // ep_size
        self.expert_start = ep_rank * self.experts_per_rank
        self.expert_end = self.expert_start + self.experts_per_rank

        # Gate is replicated within tp_group (attention TP produces
        # replicated input, so identical routing on every TP peer).
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        # Experts sharded across ep_group.
        self.experts = nn.ModuleList(
            [RefSwiGLU_MLP(hidden, intermediate) for _ in range(self.experts_per_rank)]
        )

    def weight_loader(
        self,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        self.gate.weight.data.copy_(gate_weight)
        for local_e in range(self.experts_per_rank):
            global_e = self.expert_start + local_e
            self.experts[local_e].gate_proj.weight.data.copy_(expert_gate_weights[global_e])
            self.experts[local_e].up_proj.weight.data.copy_(expert_up_weights[global_e])
            self.experts[local_e].down_proj.weight.data.copy_(expert_down_weights[global_e])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape                    # [B, T, H]
        x_flat = x.reshape(-1, self.hidden)         # [N_tp, H] replicated within tp_group
        N_tp = x_flat.shape[0]
        assert N_tp % self.tp_size == 0, (
            f"N_tp={N_tp} must be divisible by tp_size={self.tp_size} for within-TP striping"
        )
        local_N = N_tp // self.tp_size

        # ================ Phase 0: stripe within TP group ================
        # Every rank in this TP group has the same [N_tp, H] tensor;
        # we take a distinct 1/tp_size slice so MoE work isn't tp_size×
        # redundant across the TP group.
        local_start = self.tp_rank * local_N
        local_end = local_start + local_N
        local_x = x_flat[local_start:local_end]     # [local_N, H]

        # ================ Phase 1: local router on local_x ================
        router_logits = self.gate(local_x)          # [local_N, num_experts]
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(local_x.dtype)

        # ================ Phase 2: local argsort by GLOBAL expert_id ================
        top_k_experts_flat = top_k_experts.reshape(-1)                                # [local_N * k]
        top_k_weights_flat = top_k_weights.reshape(-1)                                # [local_N * k]
        sorted_expert_ids, sort_perm = torch.sort(top_k_experts_flat, stable=True)
        local_token_ids = torch.arange(local_N, device=x.device).repeat_interleave(self.top_k)
        sorted_local_token_ids = local_token_ids[sort_perm]                           # [local_N * k]
        sorted_weights = top_k_weights_flat[sort_perm]                                # [local_N * k]
        sorted_x = local_x[sorted_local_token_ids]                                    # [local_N * k, H]

        # ================ Phase 3: per-EP-destination-rank splits ================
        dest_ranks = sorted_expert_ids // self.experts_per_rank
        input_split_sizes = torch.bincount(dest_ranks, minlength=self.ep_size).to(torch.long)

        # ================ Phase 4: negotiate output splits over EP group ================
        output_split_sizes = torch.empty(self.ep_size, dtype=torch.long, device=x.device)
        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)
        input_splits_list = input_split_sizes.tolist()
        output_splits_list = output_split_sizes.tolist()

        # ================ Phase 5: DISPATCH across EP group (crosses TP-group boundaries) ================
        received_x = all_to_all_variable(
            sorted_x, input_splits_list, output_splits_list, group=self.ep_group,
        )
        received_expert_ids = all_to_all_variable(
            sorted_expert_ids.contiguous(), input_splits_list, output_splits_list, group=self.ep_group,
        )

        # ================ Phase 6: local re-argsort by LOCAL expert_id ================
        received_local_ids = received_expert_ids - self.expert_start
        sorted_local_ids, local_sort_perm = torch.sort(received_local_ids, stable=True)
        local_sorted_x = received_x[local_sort_perm]
        local_counts = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
        local_offsets = F.pad(local_counts.cumsum(0), (1, 0))

        # ================ Phase 7: local expert compute ================
        local_expert_out = torch.empty_like(local_sorted_x)
        for local_e in range(self.experts_per_rank):
            s = local_offsets[local_e].item()
            e = local_offsets[local_e + 1].item()
            if s == e:
                continue
            local_expert_out[s:e] = self.experts[local_e](local_sorted_x[s:e])

        # ================ Phase 8: reverse local sort ================
        unsorted_received_out = torch.empty_like(local_expert_out)
        unsorted_received_out[local_sort_perm] = local_expert_out

        # ================ Phase 9: COMBINE across EP group (reverse of dispatch) ================
        returned_out = all_to_all_variable(
            unsorted_received_out,
            output_splits_list, input_splits_list, group=self.ep_group,  # ← swapped
        )

        # ================ Phase 10: weight multiply + local scatter into [local_N, H] ================
        returned_out = returned_out * sorted_weights[:, None]
        local_y_flat = torch.zeros(local_N, self.hidden, device=x.device, dtype=x.dtype)
        local_y_flat.index_add_(0, sorted_local_token_ids, returned_out)

        # ================ Phase 11: ALL_GATHER within TP group → [N_tp, H] replicated within TP ================
        y_flat = torch.empty(N_tp, self.hidden, device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(y_flat, local_y_flat, group=self.tp_group)

        return y_flat.reshape(original_shape)


class HybridBlock(nn.Module):
    """One transformer block under TP-4 × DP-2 × EP-8.

    Structure (pre-norm + residual):
        h = x + attn(rmsnorm(x))
        y = h + moe(rmsnorm(h))

    - `attn` is TPGQA (TP within `tp_group`).
    - `moe` is HybridMoE (striped within tp_group, dispatched over ep_group,
      gathered within tp_group).
    - Both norms are RMSNorm applied to `[B, T, H]` replicated within tp_group
      (elementwise; no collective).

    Block-level invariant:
        { input replicated within tp_group } → forward → { output replicated within tp_group }
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        num_experts: int,
        top_k: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        ep_group: dist.ProcessGroup | None,
        norm_topk_prob: bool = True,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.attn_norm = RMSNorm(hidden, eps=rms_eps)
        self.attn = TPGQA(
            hidden, n_heads, n_kv_heads, head_dim,
            tp_size=tp_size, tp_rank=tp_rank, group=tp_group, rope_base=rope_base,
        )
        self.moe_norm = RMSNorm(hidden, eps=rms_eps)
        self.moe = HybridMoE(
            hidden, intermediate, num_experts, top_k,
            tp_size=tp_size, tp_rank=tp_rank, tp_group=tp_group,
            ep_size=ep_size, ep_rank=ep_rank, ep_group=ep_group,
            norm_topk_prob=norm_topk_prob,
        )

    def weight_loader(
        self,
        attn_norm_weight: torch.Tensor,
        q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        moe_norm_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        """Load full weights from a single-GPU RefBlock's parameters."""
        self.attn_norm.weight.data.copy_(attn_norm_weight)
        self.attn.qkv_proj.weight_loader(q_weight, "q")
        self.attn.qkv_proj.weight_loader(k_weight, "k")
        self.attn.qkv_proj.weight_loader(v_weight, "v")
        self.attn.o_proj.weight_loader(o_weight)
        self.moe_norm.weight.data.copy_(moe_norm_weight)
        self.moe.weight_loader(
            gate_weight, expert_gate_weights, expert_up_weights, expert_down_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        y = h + self.moe(self.moe_norm(h))
        return y
