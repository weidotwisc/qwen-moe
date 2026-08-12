"""Reference solution for ex05a — Naive MoE with per-expert loop.

Same math as bootcamp/ref/moe.py::RefSparseMoE (that's the "single-GPU
oracle" the tests compare against). This module exists so `USE_REFERENCE=1`
picks up a version keyed by the ex05 class name and structure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


class NaiveSparseMoE(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [RefSwiGLU_MLP(hidden, intermediate) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])

        # Router: fp32 softmax for numerical stability, then top-k.
        router_logits = self.gate(x_flat)                                     # [N, E]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x_flat.dtype)

        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            token_idx, k_idx = torch.where(selected_experts == e)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[e](x_flat[token_idx])
            expert_out = expert_out * routing_weights[token_idx, k_idx, None]
            output.index_add_(0, token_idx, expert_out)

        return output.reshape(original_shape)
