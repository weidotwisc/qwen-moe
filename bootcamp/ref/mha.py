"""Single-GPU multi-head attention with RoPE. No KV cache, no paging, no q_norm/k_norm.

Separate q_proj / k_proj / v_proj / o_proj (matches how HF safetensors store
attention weights). The TP version in ex03 will *fuse* q_proj + k_proj + v_proj
into a single packed projection, and the test loads the reference's three
separate weights into that packed layer one at a time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bootcamp.rope import apply_rope, build_rope_cache


class RefMHA(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden]
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        cos, sin = build_rope_cache(
            T, self.head_dim, base=self.rope_base, device=x.device, dtype=x.dtype
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # SDPA wants [B, H, T, D]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)
