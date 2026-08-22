"""Exercise 5a — Naive MoE (single-GPU, no parallelism).

Fill in `NaiveSparseMoE`. This is the "MoE hello world" — no
parallelism yet, just the sparse routing math + per-expert loop. Ex05b
introduces the permuted-grouped layout that a fused Triton kernel or
an EP dispatch would use; Ex06 adds all-to-all comm across ranks.

The math:

1. **Router**: linear projection `gate: hidden → num_experts`, then
   `softmax(dim=-1)` (in fp32 for numerical stability), then
   `topk(dim=-1, k=top_k)` to pick each token's k experts.
2. **Optional weight renormalization**: divide the top-k weights by
   their sum so they add to 1 (`norm_topk_prob`).
3. **Per-expert compute + weighted combine**: for each expert e,
   gather the tokens routed to it, run through the expert's SwiGLU
   MLP, weight the output by that token's routing score for expert e,
   and scatter-add into the final output.
"""

from __future__ import annotations
from turtle import forward

import torch
import torch.nn.functional as F
from torch import nn

#from bootcamp.ref.mlp import RefSwiGLU_MLP

class SwiGLU_MLP(nn.Module):
    def __init__(
        self,
        hidden:int,
        intermediate:int
    ):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=hidden, out_features=intermediate, bias=False) # for MoE FFN, the bias is false
        self.up_proj = nn.Linear(in_features=hidden, out_features=intermediate, bias=False)
        self.down_proj = nn.Linear(in_features=intermediate, out_features=hidden, bias=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class NaiveSparseMoE(nn.Module):
    """Single-GPU sparse MoE with per-expert loop.

    Args:
        hidden: input/output dim (residual stream width).
        intermediate: MLP intermediate dim per expert.
        num_experts: total experts in this layer.
        top_k: how many experts each token uses.
        norm_topk_prob: if True, renormalize top-k weights to sum to 1.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        # TODO(you):
        # 1. Store hidden, intermediate, num_experts, top_k, norm_topk_prob on self.
        # 2. self.gate = nn.Linear(hidden, num_experts, bias=False)
        #    — the "router" projection.
        # 3. self.experts = nn.ModuleList([
        #        RefSwiGLU_MLP(hidden, intermediate) for _ in range(num_experts)
        #    ])
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden, num_experts,bias=False) # the router layer, bug fix bias=False
        self.experts = nn.ModuleList(SwiGLU_MLP(intermediate=intermediate, hidden=hidden) for _ in range(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse-routed MoE forward.

        Args:
            x: [B, T, H] or [N, H] input.  H = hidden dim of the residual stream.

        Returns:
            Tensor of the same shape as x.
        """
        original_shape = x.shape # B,T,D
        x_flat = x.reshape(-1, original_shape[-1]) # B*T, D
        N = x_flat.shape[0] # N = B*T
        y_flat = torch.zeros_like(x_flat) # B*T, D
        # TODO(you):
        # 1. router_logits = self.gate(x_flat)                             # [N, num_experts]
        # 2. routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        #    Softmax in fp32 is the industry convention for numerical stability
        #    — the small-magnitude expert logits + subsequent normalization are
        #    sensitive to bf16 rounding.
        # 3. routing_weights, selected_experts = torch.topk(
        #        routing_weights, self.top_k, dim=-1
        #    )                                                              # both [N, top_k]
        # 4. if self.norm_topk_prob:
        #        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # 5. Cast routing_weights back to x_flat.dtype (fp32 → bf16 or whatever).
        # 6. output = torch.zeros_like(x_flat)
        # 7. For each expert e in range(self.num_experts):
        #        token_idx, k_idx = torch.where(selected_experts == e)     # both [num_tokens_for_e]
        #        if token_idx.numel() == 0: continue
        #        expert_out = self.experts[e](x_flat[token_idx])           # [num_tokens_for_e, H]
        #        expert_out = expert_out * routing_weights[token_idx, k_idx, None]
        #        output.index_add_(0, token_idx, expert_out)
        # 8. return output.reshape(original_shape)
        
        # weiz step 1 build up router
        router_logits = self.gate(x_flat) # N x num_expert
        top_k_weights, top_k_experts = torch.topk(router_logits, k=self.top_k, dim=-1) # both N x top_k
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1) 
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(dim=-1, index=top_k_experts)
        # weiz step 2 loop over expert
        for expert_id in range(self.num_experts):
            token_ids, weight_indices = torch.where(top_k_experts==expert_id) # both are list of size t, which is how many tokens correspond to this expert
            if len(token_ids) == 0:
                continue
            tokens = x_flat[token_ids] # (t,D)
            weights = top_k_weights[token_ids, weight_indices] # (t,)
            tokens_raw_projection = self.experts[expert_id](tokens) # (t,D)
            expert_tokens_contribution = tokens_raw_projection * weights[:,None] # (t,D)
            y_flat.index_add_(dim=0, index=token_ids,source=expert_tokens_contribution)
        return y_flat.view(original_shape)


        
