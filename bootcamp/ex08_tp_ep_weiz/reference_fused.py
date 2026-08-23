"""Ex08 hybrid schedule — with FUSED intra+inter compute loop.

Same algorithm as reference.py, but the per-expert compute is unified into
a single loop. Intra records and received-inter records are concatenated
before the loop, sorted together, computed in one pass, then un-sorted
and split for the different downstream fates:
  - Intra outputs: multiply by intra weights, scatter locally.
  - Inter outputs: send back to originators via combine, then scatter.

This eliminates the 2× kernel-launch overhead of reference.py's separate
intra/inter compute loops, which dominated wall-clock at Qwen3-30B-A3B
dims on single-node NVLink.

## Why the fusion is safe

Both intra records and received-inter records get computed on THIS rank's
same local experts. The only downstream differences are:
  - Where the output rows go (intra: locally scattered; inter: combine send).
  - Which weights get applied (intra: local, inter: from originator via combine
    but we kept weights local, applied after combine).

So the compute step is identical; only the pre-compute concat and
post-compute split differ.
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


class HybridScheduleMoEFused(nn.Module):
    """Fused-compute variant of HybridScheduleMoE.

    Constructor + weight_loader identical to reference.py's HybridScheduleMoE.
    Only forward's compute path differs.
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
        self.my_tp_group_idx = ep_rank // tp_size

        self.gate = nn.Linear(hidden, num_experts, bias=False)
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
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden)
        N_tp = x_flat.shape[0]
        device = x.device
        dtype = x_flat.dtype

        # ================ Phase 1: local router ================
        router_logits = self.gate(x_flat)
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(dtype)

        top_k_experts_flat = top_k_experts.reshape(-1)
        top_k_weights_flat = top_k_weights.reshape(-1)
        token_ids_flat = torch.arange(N_tp, device=device).repeat_interleave(self.top_k)

        # ================ Phase 2: classify ================
        dest_ranks = top_k_experts_flat // self.experts_per_rank
        dest_tp_groups = dest_ranks // self.tp_size
        intra_mask = dest_ranks == self.ep_rank
        inter_mask = dest_tp_groups != self.my_tp_group_idx

        # ================ Phase 3a: INTRA prep (gather, but no compute yet) ================
        intra_positions = intra_mask.nonzero(as_tuple=True)[0]
        intra_token_ids = token_ids_flat[intra_positions]
        intra_local_ids = top_k_experts_flat[intra_positions] - self.expert_start
        intra_weights = top_k_weights_flat[intra_positions]
        intra_x = x_flat[intra_token_ids]                                             # [N_intra, H]
        n_intra = intra_x.shape[0]

        # ================ Phase 3b: INTER dispatch (no compute yet) ================
        inter_positions_all = inter_mask.nonzero(as_tuple=True)[0]
        my_stripe = inter_positions_all[self.tp_rank :: self.tp_size]
        stripe_token_ids = token_ids_flat[my_stripe]
        stripe_expert_ids = top_k_experts_flat[my_stripe]
        stripe_weights = top_k_weights_flat[my_stripe]

        stripe_sorted_ids, stripe_sort_perm = torch.sort(stripe_expert_ids, stable=True)
        stripe_sorted_token_ids = stripe_token_ids[stripe_sort_perm]                  # KEPT LOCAL
        stripe_sorted_weights = stripe_weights[stripe_sort_perm]                      # KEPT LOCAL
        stripe_sorted_x = x_flat[stripe_sorted_token_ids]

        stripe_dest_ranks = stripe_sorted_ids // self.experts_per_rank
        input_split_sizes = torch.bincount(stripe_dest_ranks, minlength=self.ep_size).to(torch.long)
        output_split_sizes = torch.empty(self.ep_size, dtype=torch.long, device=device)
        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)
        input_splits_list = input_split_sizes.tolist()
        output_splits_list = output_split_sizes.tolist()

        received_x = all_to_all_variable(
            stripe_sorted_x, input_splits_list, output_splits_list, group=self.ep_group,
        )
        received_expert_ids = all_to_all_variable(
            stripe_sorted_ids.contiguous(), input_splits_list, output_splits_list, group=self.ep_group,
        )
        received_local_ids = received_expert_ids - self.expert_start
        n_inter = received_x.shape[0]

        # ================ Phase 4: FUSED COMPUTE — intra + received-inter together ================
        combined_x = torch.cat([intra_x, received_x], dim=0)                          # [n_intra + n_inter, H]
        combined_local_ids = torch.cat([intra_local_ids, received_local_ids], dim=0)  # [n_intra + n_inter]

        # Sort combined by local expert id.
        combined_sorted_ids, combined_sort_perm = torch.sort(combined_local_ids, stable=True)
        combined_sorted_x = combined_x[combined_sort_perm]

        # ONE compute loop over local experts.
        combined_counts = torch.bincount(combined_sorted_ids, minlength=self.experts_per_rank)
        combined_offsets = F.pad(combined_counts.cumsum(0), (1, 0))

        combined_out = torch.empty_like(combined_sorted_x)
        for local_e in range(self.experts_per_rank):
            s = combined_offsets[local_e].item()
            e = combined_offsets[local_e + 1].item()
            if s == e:
                continue
            combined_out[s:e] = self.experts[local_e](combined_sorted_x[s:e])

        # Un-permute back to (intra, received) concat order.
        unsorted_out = torch.empty_like(combined_out)
        unsorted_out[combined_sort_perm] = combined_out

        # Split: first n_intra rows are intra outputs; rest are inter outputs.
        intra_out = unsorted_out[:n_intra]
        inter_out = unsorted_out[n_intra:]

        # ================ Phase 5: INTRA scatter (local) ================
        partial_output = torch.zeros(N_tp, self.hidden, device=device, dtype=dtype)
        if n_intra > 0:
            intra_weighted = intra_out * intra_weights[:, None]
            partial_output.index_add_(0, intra_token_ids, intra_weighted)

        # ================ Phase 6: INTER combine (send back to originators) + scatter ================
        if n_inter > 0 or any(s > 0 for s in input_splits_list):
            returned_out = all_to_all_variable(
                inter_out,
                output_splits_list, input_splits_list, group=self.ep_group,  # swapped
            )
            if returned_out.shape[0] > 0:
                returned_weighted = returned_out * stripe_sorted_weights[:, None]
                partial_output.index_add_(0, stripe_sorted_token_ids, returned_weighted)

        # ================ Phase 7: within-TP all_reduce ================
        dist.all_reduce(partial_output, op=dist.ReduceOp.SUM, group=self.tp_group)

        return partial_output.reshape(original_shape)


class HybridScheduleBlockFused(nn.Module):
    """HybridScheduleBlock with fused MoE compute path (single per-expert loop)."""

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
        self.moe = HybridScheduleMoEFused(
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
