"""EPSparseMoE — dispatch (all_to_all × 2) variant with Ex09's fused Triton
kernel replacing the per-expert Python loop.

Subclasses reference.EPSparseMoE and overrides ONLY the local expert compute;
routing, dispatch, combine, all_gather are all inherited semantics — the code
below is copy-pasted from the parent forward so we can swap the one stanza
without contorting the class hierarchy.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from bootcamp.ex00_dist_primer.solution import all_to_all_variable
from bootcamp.ex06_ep.reference import EPSparseMoE as DispatchEPSparseMoE
from bootcamp.ex09_fused_moe.reference import fused_moe_forward, pack_expert_weights


class EPSparseMoEDispatchFused(DispatchEPSparseMoE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._packed: dict[str, torch.Tensor] | None = None

    def _ensure_packed(self) -> dict[str, torch.Tensor]:
        if self._packed is None:
            self._packed = pack_expert_weights(self.experts)
        return self._packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        N = x_flat.shape[0]
        assert N % self.ep_size == 0
        local_N = N // self.ep_size

        # Phase 0
        local_start = self.ep_rank * local_N
        local_end = local_start + local_N
        local_x = x_flat[local_start:local_end]

        # Phase 1
        router_logits = self.gate(local_x)
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(local_x.dtype)

        # Phase 2
        top_k_experts_flat = top_k_experts.reshape(-1)
        top_k_weights_flat = top_k_weights.reshape(-1)
        sorted_expert_ids, sort_perm = torch.sort(top_k_experts_flat, stable=True)
        local_token_ids = torch.arange(local_N, device=x.device).repeat_interleave(self.top_k)
        sorted_local_token_ids = local_token_ids[sort_perm]
        sorted_weights = top_k_weights_flat[sort_perm]
        local_x_permuted = local_x[sorted_local_token_ids]

        # Phase 3
        dest_ranks = sorted_expert_ids // self.experts_per_rank
        input_split_sizes = torch.bincount(dest_ranks, minlength=self.ep_size).to(torch.long)

        # Phase 4
        all_input_splits = torch.zeros(
            self.ep_size * self.ep_size, dtype=torch.long, device=x.device
        )
        dist.all_gather_into_tensor(all_input_splits, input_split_sizes, group=self.group)
        all_input_splits = all_input_splits.view(self.ep_size, self.ep_size)
        output_split_sizes = all_input_splits[:, self.ep_rank].contiguous()
        input_splits_list = input_split_sizes.tolist()
        output_splits_list = output_split_sizes.tolist()

        # Phase 5: DISPATCH
        received_x = all_to_all_variable(
            local_x_permuted, input_splits_list, output_splits_list, group=self.group,
        )
        received_expert_ids = all_to_all_variable(
            sorted_expert_ids.contiguous(), input_splits_list, output_splits_list,
            group=self.group,
        )

        # Phase 6: local re-argsort
        received_local_ids = received_expert_ids - self.expert_start
        sorted_local_ids, local_sort_perm = torch.sort(received_local_ids, stable=True)
        local_sorted_x = received_x[local_sort_perm]
        local_counts = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
        local_offsets = F.pad(local_counts.cumsum(0), (1, 0)).to(torch.int64)

        # Phase 7 (FUSED): single grouped-GEMM launch instead of Python loop.
        packed = self._ensure_packed()
        local_expert_out = fused_moe_forward(
            local_sorted_x, local_offsets,
            packed["W_gate"], packed["W_up"], packed["W_down"],
        )

        # Phase 8: reverse local sort
        unsorted_received_out = torch.empty_like(local_expert_out)
        unsorted_received_out[local_sort_perm] = local_expert_out

        # Phase 9: COMBINE
        returned_out = all_to_all_variable(
            unsorted_received_out,
            output_splits_list,
            input_splits_list,
            group=self.group,
        )

        # Phase 10: weight + scatter
        returned_out = returned_out * sorted_weights[:, None]
        local_y_flat = torch.zeros(local_N, self.hidden, device=x.device, dtype=x.dtype)
        local_y_flat.index_add_(0, sorted_local_token_ids, returned_out)

        # Phase 11: all_gather full output
        y_flat = torch.empty(N, self.hidden, device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(y_flat, local_y_flat, group=self.group)

        return y_flat.reshape(original_shape)
