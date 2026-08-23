"""Reference implementation for Ex08 — Wei's hybrid intra-lean/inter-dispatch schedule.

The observation motivating this design (from the Ex07 discussion):

    Under TP × DP × EP composition (case 3), MoE routing records
    partition into two classes:
      - INTRA-TP-group records: dest expert is on a rank in the SAME
        TP group as this rank. The token is already replicated across
        the TP group (from attention TP AR). No cross-network
        transport needed.
      - INTER-TP-group records: dest expert is on a rank in a DIFFERENT
        TP group. Token must be sent via dispatch.

    Ex07's canonical schedule dispatches ALL records (redundantly for
    intra-TP-group ones). Wei's hybrid schedule uses:
      - Lean pattern for intra records (filter + local compute + within-TP AR).
      - Dispatch pattern for inter records (all_to_all_v across EP group
        with zero counts for intra-group destinations).

    Both intra and inter contributions accumulate into the same
    partial_output; one final within-TP all_reduce distributes the
    complete result across the TP group.

Bandwidth savings vs Ex07 canonical: ~37% at case (3) uniform routing.

## Contract

Input contract identical to Ex07's HybridBlock:
    x on rank r = [B, T, H] — this rank's TP group's half of the batch,
    replicated across the group's ranks.
Output:
    y on rank r = [B, T, H] — same shape, replicated within TP group.

## The two-branch schedule

  Phase 1: local router on replicated input (redundant across TP group, cheap).
  Phase 2: classify each record as intra-TP or inter-TP.
  Phase 3: INTRA branch — each rank filters to (dest == my_rank), local compute,
           partial_output.index_add_.
  Phase 4: INTER branch — each rank strides its 1/tp_size of inter records,
           all_to_all_v (across EP, intra dests are zero-count), local compute,
           all_to_all_v combine, partial_output.index_add_.
  Phase 5: all_reduce partial_output within TP group.
  Return.

## Cross-group composition proof structure

  { input replicated within tp_group }
    → attn TP-4 (with attn AR within tp_group)
    → HybridScheduleMoE (with 2 EP all_to_all_v + 1 TP all_reduce)
    → output replicated within tp_group

  All collectives are on fixed schedule; group handles are explicit.
  Sub-group interleaving (tp_group's AR interleaved with ep_group's
  all_to_all_v) preserves the composition theorem from Ex07.
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


class HybridScheduleMoE(nn.Module):
    """MoE with intra-TP-lean + inter-TP-dispatch hybrid schedule.

    Args mirror Ex07's HybridMoE — same TP + EP parameters, same weight-loader
    contract.
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

        # Which TP group I belong to (assumes contiguous TP groupings).
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
        original_shape = x.shape                            # [B, T, H]
        x_flat = x.reshape(-1, self.hidden)                 # [N_tp, H] replicated in tp_group
        N_tp = x_flat.shape[0]

        # ================ Phase 1: local router ================
        router_logits = self.gate(x_flat)                   # [N_tp, num_experts]
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(x_flat.dtype)

        top_k_experts_flat = top_k_experts.reshape(-1)                                # [N_tp * k]
        top_k_weights_flat = top_k_weights.reshape(-1)                                # [N_tp * k]
        token_ids_flat = torch.arange(N_tp, device=x.device).repeat_interleave(self.top_k)

        # ================ Phase 2: classify records ================
        # dest_rank = expert_id // experts_per_rank; dest_tp_group = dest_rank // tp_size.
        dest_ranks = top_k_experts_flat // self.experts_per_rank
        dest_tp_groups = dest_ranks // self.tp_size
        is_intra = dest_tp_groups == self.my_tp_group_idx
        # NOTE: dest_ranks == self.ep_rank implies is_intra automatically (same TP group).
        # So intra_mask can be derived as just (dest_ranks == self.ep_rank).
        intra_mask = dest_ranks == self.ep_rank
        inter_mask = ~is_intra

        partial_output = torch.zeros(N_tp, self.hidden, device=x.device, dtype=x_flat.dtype)

        # ================ Phase 3: INTRA branch — local, no dispatch ================
        # Every rank in my TP group filters to its OWN dest_rank. Disjoint filters
        # cover all intra records collectively (through the within-TP AR at Phase 5).
        if intra_mask.any():
            intra_positions = intra_mask.nonzero(as_tuple=True)[0]
            intra_token_ids = token_ids_flat[intra_positions]
            intra_expert_ids = top_k_experts_flat[intra_positions]
            intra_weights = top_k_weights_flat[intra_positions]

            # Sort by local expert_id for contiguous per-expert compute
            intra_local_ids = intra_expert_ids - self.expert_start
            sorted_local_ids, sort_perm = torch.sort(intra_local_ids, stable=True)
            sorted_token_ids = intra_token_ids[sort_perm]
            sorted_weights = intra_weights[sort_perm]
            sorted_x = x_flat[sorted_token_ids]

            local_counts = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
            local_offsets = F.pad(local_counts.cumsum(0), (1, 0))

            intra_expert_out = torch.empty_like(sorted_x)
            for local_e in range(self.experts_per_rank):
                s = local_offsets[local_e].item()
                e = local_offsets[local_e + 1].item()
                if s == e:
                    continue
                intra_expert_out[s:e] = self.experts[local_e](sorted_x[s:e])
            intra_expert_out = intra_expert_out * sorted_weights[:, None]

            partial_output.index_add_(0, sorted_token_ids, intra_expert_out)

        # ================ Phase 4: INTER branch — stripe within TP + dispatch across EP ================
        # Positions of inter records:
        inter_positions_all = inter_mask.nonzero(as_tuple=True)[0]                    # [num_inter]
        # Stripe: this rank takes 1/tp_size of the inter records.
        my_stripe = inter_positions_all[self.tp_rank :: self.tp_size]

        stripe_token_ids = token_ids_flat[my_stripe]
        stripe_expert_ids = top_k_experts_flat[my_stripe]
        stripe_weights = top_k_weights_flat[my_stripe]

        # Sort by dest expert_id so records are grouped by dest_rank
        stripe_sorted_ids, stripe_sort_perm = torch.sort(stripe_expert_ids, stable=True)
        stripe_sorted_token_ids = stripe_token_ids[stripe_sort_perm]                  # KEPT LOCAL
        stripe_sorted_weights = stripe_weights[stripe_sort_perm]                      # KEPT LOCAL
        stripe_sorted_x = x_flat[stripe_sorted_token_ids]

        # Per-EP-destination splits. Intra destinations get ZERO (since all inter
        # records have dest outside my TP group). This is the "zero-count intra"
        # trick that fakes point-to-point-only cross-group traffic through the
        # same all_to_all_v API.
        stripe_dest_ranks = stripe_sorted_ids // self.experts_per_rank
        input_split_sizes = torch.bincount(stripe_dest_ranks, minlength=self.ep_size).to(torch.long)

        # Negotiate output splits over EP group
        output_split_sizes = torch.empty(self.ep_size, dtype=torch.long, device=x.device)
        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)
        input_splits_list = input_split_sizes.tolist()
        output_splits_list = output_split_sizes.tolist()

        # DISPATCH — token bytes cross TP-group boundary (intra-group entries are zero-size)
        received_x = all_to_all_variable(
            stripe_sorted_x, input_splits_list, output_splits_list, group=self.ep_group,
        )
        received_expert_ids = all_to_all_variable(
            stripe_sorted_ids.contiguous(), input_splits_list, output_splits_list, group=self.ep_group,
        )

        # Local re-argsort by local expert_id + compute
        if received_x.shape[0] > 0:
            received_local_ids = received_expert_ids - self.expert_start
            sorted_local_ids, local_sort_perm = torch.sort(received_local_ids, stable=True)
            local_sorted_x = received_x[local_sort_perm]
            local_counts = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
            local_offsets = F.pad(local_counts.cumsum(0), (1, 0))

            local_expert_out = torch.empty_like(local_sorted_x)
            for local_e in range(self.experts_per_rank):
                s = local_offsets[local_e].item()
                e = local_offsets[local_e + 1].item()
                if s == e:
                    continue
                local_expert_out[s:e] = self.experts[local_e](local_sorted_x[s:e])

            unsorted_received_out = torch.empty_like(local_expert_out)
            unsorted_received_out[local_sort_perm] = local_expert_out
        else:
            unsorted_received_out = torch.empty(0, self.hidden, device=x.device, dtype=x_flat.dtype)

        # COMBINE — send results back
        returned_out = all_to_all_variable(
            unsorted_received_out,
            output_splits_list, input_splits_list, group=self.ep_group,  # ← swapped
        )

        # Multiply by weights (kept local on the originator) and scatter into partial_output
        if returned_out.shape[0] > 0:
            returned_out = returned_out * stripe_sorted_weights[:, None]
            partial_output.index_add_(0, stripe_sorted_token_ids, returned_out)

        # ================ Phase 5: all_reduce within TP GROUP ================
        # Sums:
        #   - Intra contributions across all TP-group ranks (each contributed to disjoint experts).
        #   - Inter contributions across all TP-group ranks (each striped a disjoint 1/tp_size of inter records).
        # Result: every rank in TP group has the complete MoE output for its group's tokens.
        dist.all_reduce(partial_output, op=dist.ReduceOp.SUM, group=self.tp_group)

        return partial_output.reshape(original_shape)


class HybridScheduleBlock(nn.Module):
    """One transformer block using Wei's hybrid intra-lean/inter-dispatch MoE.

    Attention path identical to Ex07's HybridBlock — TP-4 within tp_group.
    Only the MoE differs: HybridScheduleMoE replaces HybridMoE.
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
        self.moe = HybridScheduleMoE(
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
