"""Reference implementation for Ex06 — Expert Parallelism (EP).

Working EP implementation. Same interface as `solution.EPSparseMoE`.
Loaded when the test environment variable `USE_REFERENCE=1` is set,
so the test harness can validate the reference itself before Wei fills
in the solution.

See [README.md](README.md) for the 11-phase pipeline description.
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
        assert num_experts % ep_size == 0, (
            f"num_experts={num_experts} not divisible by ep_size={ep_size}"
        )
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

        # REPLICATED across ranks — every rank owns the full gate weight.
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        # SHARDED — rank r only holds experts_per_rank MLPs.
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
        x_flat = x.reshape(-1, original_shape[-1])
        N = x_flat.shape[0]
        assert N % self.ep_size == 0, (
            f"N={N} must be divisible by ep_size={self.ep_size}"
        )
        local_N = N // self.ep_size

        # ================ Phase 0: local token partition ================
        local_start = self.ep_rank * local_N
        local_end = local_start + local_N
        local_x = x_flat[local_start:local_end]                              # [local_N, H]

        # ================ Phase 1: local router ================
        router_logits = self.gate(local_x)                                   # [local_N, E]
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)                 # [local_N, k]
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(local_x.dtype)

        # ================ Phase 2: local argsort by GLOBAL expert_id ================
        top_k_experts_flat = top_k_experts.reshape(-1)                       # [local_N * k]
        top_k_weights_flat = top_k_weights.reshape(-1)                       # [local_N * k]
        sorted_expert_ids, sort_perm = torch.sort(top_k_experts_flat, stable=True)
        local_token_ids = torch.arange(local_N, device=x.device).repeat_interleave(self.top_k)
        sorted_local_token_ids = local_token_ids[sort_perm]                  # [local_N * k]
        sorted_weights = top_k_weights_flat[sort_perm]                       # [local_N * k]
        local_x_permuted = local_x[sorted_local_token_ids]                   # [local_N * k, H]

        # ================ Phase 3: compute per-destination-rank splits ================
        dest_ranks = sorted_expert_ids // self.experts_per_rank              # [local_N * k]
        input_split_sizes = torch.bincount(dest_ranks, minlength=self.ep_size).to(torch.long)  # [ep_size]

        # ================ Phase 4: negotiate output_split_sizes ================
        all_input_splits = torch.zeros(
            self.ep_size * self.ep_size, dtype=torch.long, device=x.device
        )
        dist.all_gather_into_tensor(all_input_splits, input_split_sizes, group=self.group)
        all_input_splits = all_input_splits.view(self.ep_size, self.ep_size)
        output_split_sizes = all_input_splits[:, self.ep_rank].contiguous()   # [ep_size]

        input_splits_list = input_split_sizes.tolist()
        output_splits_list = output_split_sizes.tolist()

        # ================ Phase 5: DISPATCH — all_to_all_variable × 2 ================
        received_x = all_to_all_variable(
            local_x_permuted, input_splits_list, output_splits_list, group=self.group,
        )                                                                    # [Nk_recv, H]
        received_expert_ids = all_to_all_variable(
            sorted_expert_ids.contiguous(), input_splits_list, output_splits_list, group=self.group,
        )                                                                    # [Nk_recv]

        # ================ Phase 6: local re-argsort by LOCAL expert_id ================
        received_local_ids = received_expert_ids - self.expert_start          # [0, experts_per_rank)
        sorted_local_ids, local_sort_perm = torch.sort(received_local_ids, stable=True)
        local_sorted_x = received_x[local_sort_perm]                          # contiguous per local expert
        local_counts = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
        local_offsets = F.pad(local_counts.cumsum(0), (1, 0))                 # [experts_per_rank + 1]

        # ================ Phase 7: local expert compute ================
        local_expert_out = torch.empty_like(local_sorted_x)
        for local_e in range(self.experts_per_rank):
            s = local_offsets[local_e].item()
            e = local_offsets[local_e + 1].item()
            if s == e:
                continue
            local_expert_out[s:e] = self.experts[local_e](local_sorted_x[s:e])

        # ================ Phase 8: reverse the local sort ================
        unsorted_received_out = torch.empty_like(local_expert_out)
        unsorted_received_out[local_sort_perm] = local_expert_out             # scatter to original position

        # ================ Phase 9: COMBINE — reverse dispatch ================
        returned_out = all_to_all_variable(
            unsorted_received_out,
            output_splits_list,     # ← swapped: what I received on dispatch, I now send on combine
            input_splits_list,      # ← swapped: what I sent on dispatch, I now receive on combine
            group=self.group,
        )                                                                     # [local_N * k, H]

        # ================ Phase 10: weight multiply + local scatter ================
        returned_out = returned_out * sorted_weights[:, None]                 # one big multiply
        local_y_flat = torch.zeros(local_N, self.hidden, device=x.device, dtype=x.dtype)
        local_y_flat.index_add_(0, sorted_local_token_ids, returned_out)      # one big scatter

        # ================ Phase 11: all_gather to reassemble full output ================
        y_flat = torch.empty(N, self.hidden, device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(y_flat, local_y_flat, group=self.group)

        return y_flat.reshape(original_shape)
