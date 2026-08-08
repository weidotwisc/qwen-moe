"""Single-GPU sparse MoE with top-k routing and per-expert loop.

Matches the shape and math of Qwen3's SparseMoeBlock:
    - Linear gate → softmax (in fp32) → top-k → optional renorm.
    - Per-expert forward: gather routed tokens, run through a SwiGLU MLP,
      weight by routing scores, scatter-sum back into the output tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


class RefSparseMoE(nn.Module):
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
        # Accept [B, T, H] or [N, H]; return same shape.
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])

        # Router
        router_logits = self.gate(x_flat)  # [N, E]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # both [N, K]
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        # Per-expert compute + combine
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            token_idx, k_idx = torch.where(selected_experts == e)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[e](x_flat[token_idx])
            out.index_add_(0, token_idx, expert_out * routing_weights[token_idx, k_idx, None])

        return out.reshape(original_shape)
