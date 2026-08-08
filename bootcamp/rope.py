"""Minimal RoPE helpers used by both the reference and student solutions.

Half-rotation formulation (same as HF's Qwen / Llama implementations):
    rope(x, pos) = x * cos(theta_pos) + rotate_half(x) * sin(theta_pos)
    where theta_j = pos / base^(2j / head_dim).

These helpers are dtype-preserving and work on tensors of shape
[..., seq, n_heads, head_dim] — i.e. RoPE is applied to the last two axes.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape [seq_len, head_dim] in the given dtype."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., seq, n_heads, head_dim]. cos/sin: [seq, head_dim]."""
    cos = cos.unsqueeze(-2)  # [seq, 1, head_dim]
    sin = sin.unsqueeze(-2)
    return x * cos + rotate_half(x) * sin
