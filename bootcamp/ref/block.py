"""Single-GPU reference transformer block for Ex07's test oracle.

Composes RefGQA (attention) + RefSparseMoE + RMSNorm + residuals — a
canonical pre-norm MoE block matching Qwen3-30B-A3B's layer structure
(minus QK-norm, which is a Qwen3-specific attention detail we skip).

Contract: `x: [B, T, H]` → `y: [B, T, H]`, no parallelism.
Used as the test oracle for HybridBlock's TP-4 × DP-2 × EP-8 composition.
"""

from __future__ import annotations

import torch
from torch import nn

from bootcamp.ref.gqa import RefGQA
from bootcamp.ref.moe import RefSparseMoE


class RMSNorm(nn.Module):
    """Root-mean-square layer norm — no bias, no mean subtraction.

    Matches HF's / Qwen3's convention: `y = weight * x / sqrt(mean(x^2) + eps)`.
    """

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean-square in fp32 for numerical stability, cast back.
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        out = xf * torch.rsqrt(var + self.eps)
        return (out.to(dtype)) * self.weight


class RefBlock(nn.Module):
    """Single-GPU MoE transformer block, pre-norm structure.

    ```
    h = x + attn(rmsnorm(x))
    y = h + moe(rmsnorm(h))
    ```

    Skips QK-norm to match Ex04's TPGQA. All Qwen3-30B-A3B dims otherwise.
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
        norm_topk_prob: bool = True,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden, eps=rms_eps)
        self.attn = RefGQA(hidden, n_heads, n_kv_heads, head_dim, rope_base=rope_base)
        self.moe_norm = RMSNorm(hidden, eps=rms_eps)
        self.moe = RefSparseMoE(
            hidden, intermediate, num_experts, top_k, norm_topk_prob=norm_topk_prob
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        y = h + self.moe(self.moe_norm(h))
        return y
