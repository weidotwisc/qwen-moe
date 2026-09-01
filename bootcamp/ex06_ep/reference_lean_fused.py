"""EPSparseMoE — lean (single all_reduce) variant with Ex09's fused Triton
kernel replacing the per-expert Python loop.

Subclasses reference_lean.EPSparseMoE and overrides ONLY the forward step
that runs the per-expert compute; all routing / weighting / all_reduce logic
is inherited unchanged. The one-line-composition boundary is identical to
Ex10's: `expert_out[s:e] = self.experts[local_e](sorted_x[s:e])` in a Python
loop becomes `expert_out = fused_moe_forward(sorted_x, offsets, W_gate,
W_up, W_down)` from Ex09.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from bootcamp.ex06_ep.reference_lean import EPSparseMoE as LeanEPSparseMoE
from bootcamp.ex09_fused_moe.reference import fused_moe_forward, pack_expert_weights


class EPSparseMoELeanFused(LeanEPSparseMoE):
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

        # Phase 1: local router (identical to parent).
        router_logits = self.gate(x_flat)
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(x_flat.dtype)

        # Phase 2: filter to LOCAL experts only (this is what "lean" saves).
        top_k_experts_flat = top_k_experts.reshape(-1)
        top_k_weights_flat = top_k_weights.reshape(-1)
        local_token_ids = (
            torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        )

        local_mask = (
            (top_k_experts_flat >= self.expert_start)
            & (top_k_experts_flat < self.expert_start + self.experts_per_rank)
        )
        local_expert_ids = top_k_experts_flat[local_mask] - self.expert_start
        local_weights = top_k_weights_flat[local_mask]
        local_token_ids = local_token_ids[local_mask]

        # Phase 3: sort by local expert id.
        sorted_local_ids, sort_perm = torch.sort(local_expert_ids, stable=True)
        sorted_weights = local_weights[sort_perm]
        sorted_token_ids = local_token_ids[sort_perm]
        sorted_x = x_flat[sorted_token_ids]

        # Phase 4: build offsets.
        local_counts = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
        local_offsets = F.pad(local_counts.cumsum(0), (1, 0)).to(torch.int64)

        # Phase 5 (FUSED): single grouped-GEMM launch instead of Python loop.
        packed = self._ensure_packed()
        expert_out = fused_moe_forward(
            sorted_x, local_offsets,
            packed["W_gate"], packed["W_up"], packed["W_down"],
        )
        expert_out = expert_out * sorted_weights[:, None]

        # Phase 6: build partial output (zeros on tokens not owned by this rank's experts).
        partial_output = torch.zeros(N, self.hidden, device=x.device, dtype=x.dtype)
        partial_output.index_add_(0, sorted_token_ids, expert_out)

        # Phase 7: single all_reduce assembles full output on every rank.
        dist.all_reduce(partial_output, op=dist.ReduceOp.SUM, group=self.group)

        return partial_output.reshape(original_shape)
