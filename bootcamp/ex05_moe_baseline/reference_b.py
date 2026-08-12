"""Reference solution for ex05b — Permuted MoE with grouped compute."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


class PermutedSparseMoE(nn.Module):
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
        N = x_flat.shape[0]

        # ------- router (identical to naive) -------
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x_flat.dtype)

        # ------- flatten routing to N*top_k records -------
        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)   # [N*K]
        expert_ids = selected_experts.reshape(-1)                                     # [N*K]
        weights = routing_weights.reshape(-1)                                         # [N*K]

        # ------- argsort by expert to compute permutation -------
        permutation = torch.argsort(expert_ids)
        sorted_expert_ids = expert_ids[permutation]
        sorted_token_ids = token_ids[permutation]
        sorted_weights = weights[permutation]

        # ------- expert offsets -------
        counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            counts.cumsum(0),
        ])                                                                            # [num_experts + 1]

        # ------- gather sorted input -------
        sorted_x = x_flat[sorted_token_ids]                                           # [N*K, hidden]

        # ------- grouped per-expert compute -------
        sorted_out = torch.empty_like(sorted_x)
        for e in range(self.num_experts):
            start, end = int(offsets[e]), int(offsets[e + 1])
            if start == end:
                continue
            sorted_out[start:end] = self.experts[e](sorted_x[start:end])

        # ------- weight + unpermute + combine -------
        sorted_out = sorted_out * sorted_weights.unsqueeze(-1)
        output = torch.zeros_like(x_flat)
        output.index_add_(0, sorted_token_ids, sorted_out)

        return output.reshape(original_shape)
