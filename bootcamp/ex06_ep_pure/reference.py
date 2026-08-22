"""Reference implementation for Ex06 pure EP — DP-partitioned inputs.

Each rank owns a distinct 1/ep_size share of the batch tokens. Dispatch
is structurally necessary — no rank has the full data, so tokens must
cross rank boundaries to reach their assigned experts.

Compared to ex06_ep/reference.py, this variant strips out:
- Phase 0 (input slicing) — input is already local to each rank.
- Phase 11 (final all_gather) — output stays partitioned per rank.

Otherwise the algorithm is identical: argsort → dispatch → local compute →
combine → weight-multiply + scatter.

Contract:
    Input:  local_x on rank r = [local_N, H] (or [B, T, H] with local_N = B*T).
            Content is distinct across ranks.
    Output: local_y on rank r = [local_N, H] — output for THIS rank's tokens only.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex00_dist_primer.solution import all_to_all_variable
from bootcamp.ref.mlp import RefSwiGLU_MLP


class EPSparseMoE(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        group: dist.ProcessGroup | None = None,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        assert num_experts % ep_size == 0
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.group = group
        self.norm_topk_prob = norm_topk_prob
        self.experts_per_rank = num_experts // ep_size
        self.expert_start = ep_rank * self.experts_per_rank
        self.expert_end = self.expert_start + self.experts_per_rank

        # REPLICATED: every rank owns the full gate weight so the router
        # produces the same routing given identical input. In DP+EP setups
        # this is standard (gate is replicated across DP replicas).
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        # SHARDED: only this rank's experts_per_rank MLPs.
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

    def forward(self, local_x: torch.Tensor) -> torch.Tensor:
        """Pure-EP forward with DP-partitioned inputs.

        Args:
            local_x: [B, T, H] or [local_N, H] — THIS rank's share of tokens.
                     Distinct content across ranks.
        Returns:
            [B, T, H] or [local_N, H] — output for THIS rank's tokens only.
        """
        original_shape = local_x.shape
        local_x_flat = local_x.reshape(-1, self.hidden)
        local_N = local_x_flat.shape[0]

        # ================ Phase 1: local router ================
        router_logits = self.gate(local_x_flat)                              # [local_N, E]
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)                 # [local_N, k]
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(local_x_flat.dtype)

        # ================ Phase 2: local argsort by GLOBAL expert_id ================
        top_k_experts_flat = top_k_experts.reshape(-1)                       # [local_N * k]
        top_k_weights_flat = top_k_weights.reshape(-1)                       # [local_N * k]
        sorted_expert_ids, sort_perm = torch.sort(top_k_experts_flat, stable=True)
        local_token_ids = torch.arange(local_N, device=local_x.device).repeat_interleave(self.top_k)
        sorted_local_token_ids = local_token_ids[sort_perm]                  # [local_N * k]
        sorted_weights = top_k_weights_flat[sort_perm]                       # [local_N * k]
        sorted_x = local_x_flat[sorted_local_token_ids]                      # [local_N * k, H]

        # ================ Phase 3: per-destination-rank splits ================
        dest_ranks = sorted_expert_ids // self.experts_per_rank
        input_split_sizes = torch.bincount(dest_ranks, minlength=self.ep_size).to(torch.long)

        # ================ Phase 4: negotiate output splits ================
        output_split_sizes = torch.empty(self.ep_size, dtype=torch.long, device=local_x.device)
        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.group)
        input_splits_list = input_split_sizes.tolist()
        output_splits_list = output_split_sizes.tolist()

        # ================ Phase 5: DISPATCH — all_to_all_variable × 2 ================
        received_x = all_to_all_variable(
            sorted_x, input_splits_list, output_splits_list, group=self.group,
        )
        received_expert_ids = all_to_all_variable(
            sorted_expert_ids.contiguous(), input_splits_list, output_splits_list, group=self.group,
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

        # ================ Phase 9: COMBINE — reverse dispatch ================
        returned_out = all_to_all_variable(
            unsorted_received_out,
            output_splits_list, input_splits_list, group=self.group,  # ← swapped
        )

        # ================ Phase 10: weight multiply + local scatter ================
        returned_out = returned_out * sorted_weights[:, None]
        local_y_flat = torch.zeros(
            local_N, self.hidden, device=local_x.device, dtype=local_x.dtype,
        )
        local_y_flat.index_add_(0, sorted_local_token_ids, returned_out)

        # ================ Phase 11: return local output — NO all_gather ================
        return local_y_flat.reshape(original_shape)
