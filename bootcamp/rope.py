"""Minimal RoPE helpers used by both the reference and student solutions.

Half-rotation formulation (same math as Meta's original LLaMA reference
release, Qwen, Mistral, and every derivative thereof):

    rope(x, pos) = x * cos(theta_pos) + rotate_half(x) * sin(theta_pos)
    where theta_j = pos / base^(2j / head_dim).

## Layout convention: RoPE-before-transpose (older vLLM/nanovllm style)

`apply_rope` expects `x` in `[..., seq, n_heads, head_dim]` layout — i.e.
the seq axis is second-to-last, head axis is one-before-last, head_dim is
innermost. This is what you get IMMEDIATELY after a projection + head
reshape, BEFORE the transpose into SDPA-friendly `[batch, head, seq, head_dim]`.

The intended call sequence in an attention block:

    q = qkv_proj(x).view(B, T, H, D)           # [B, T, H, D]
    q = apply_rope(q, cos, sin)                 # still [B, T, H, D]
    q = q.transpose(1, 2)                       # [B, H, T, D] for SDPA

Not:

    q = q.view(B, T, H, D).transpose(1, 2)      # [B, H, T, D]
    q = apply_rope(q, cos, sin)                 # ← WON'T WORK — see below

The `cos.unsqueeze(-2)` inside `apply_rope` produces `[T, 1, D]`, which
broadcasts cleanly against `[B, T, H, D]` (T-of-cos aligns with T-of-x,
size-1 broadcasts across H, D matches D). Under `[B, H, T, D]` layout,
T-of-cos would try to align with H-of-x — non-1 dims must match, so it
either errors or produces wrong output silently.

## Why this convention (not the modern HF one)

Historical: **Meta's original LLaMA reference code (Feb 2023) applied RoPE
before the transpose.** vLLM (SOSP 2023) inherited this convention, and
nanovllm inherits from vLLM. Modern HF `modeling_qwen3_moe.py` transposes
first and uses a differently-shaped RoPE helper — cleaner reading, but
would require refactoring vLLM's fused-attention kernels (which assume
this layout at the RoPE boundary), so vLLM/nanovllm kept the older style.

We match the older convention so the bootcamp code ports cleanly back to
nanovllm-jun. For your paper's Verus spec, RoPE's semantics are
layout-invariant — this quirk is a Python-implementation detail only.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half, negate the second half, swap the two halves.
    Used inside apply_rope. Standard LLaMA-1 formulation.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape [seq_len, head_dim] in the given dtype.

    Independent of head index — cos/sin depend only on position + head_dim,
    so every rank in a TP setup builds the identical cache. No comm needed.
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # [seq_len, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)   # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to `x` in the pre-transpose layout.

    Args:
        x:   shape `[..., seq, n_heads, head_dim]` — seq at position -3, heads at -2, dim at -1.
             This is the layout AFTER `.view(B, T, H, D)` and BEFORE the SDPA
             transpose to `[B, H, T, D]`. See module docstring.
        cos, sin: shape `[seq, head_dim]` from `build_rope_cache`.

    Returns: tensor of the same shape as `x`.

    The `unsqueeze(-2)` produces `[seq, 1, head_dim]`, which broadcasts
    against `x`'s `[..., seq, n_heads, head_dim]` — the size-1 axis
    broadcasts across `n_heads` so every head rotates by the same
    seq-position-dependent angles.
    """
    cos = cos.unsqueeze(-2)   # [seq, 1, head_dim]
    sin = sin.unsqueeze(-2)   # [seq, 1, head_dim]
    return x * cos + rotate_half(x) * sin
