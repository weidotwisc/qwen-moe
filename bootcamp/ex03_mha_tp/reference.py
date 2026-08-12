"""Reference solutions for ex03.

Imports the ex01 REFERENCE (not solution) so this file works stand-alone.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex01_linear_tp.reference import ColumnParallelLinear, RowParallelLinear
from bootcamp.rope import apply_rope, build_rope_cache


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden: int,
        head_dim: int,   # nanovllm/vLLM call this `head_size`; we use `head_dim` (matches HF + Qwen3 config)
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        assert num_heads == num_kv_heads, (
            "ex03 handles MHA only (num_heads == num_kv_heads). GQA is exercise 4."
        )
        assert num_heads % tp_size == 0
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size_per_rank = (num_heads // tp_size) * head_dim
        self.kv_size_per_rank = (num_kv_heads // tp_size) * head_dim
        output_size = (num_heads + 2 * num_kv_heads) * head_dim
        super().__init__(hidden, output_size, tp_size, tp_rank, group=group)

    def weight_loader(self, full_weight: torch.Tensor, shard_id: str) -> None:  # type: ignore[override]
        if shard_id == "q":
            offset, size = 0, self.q_size_per_rank
        elif shard_id == "k":
            offset, size = self.q_size_per_rank, self.kv_size_per_rank
        elif shard_id == "v":
            offset, size = self.q_size_per_rank + self.kv_size_per_rank, self.kv_size_per_rank
        else:
            raise ValueError(f"shard_id must be 'q'|'k'|'v', got {shard_id!r}")
        # full_weight is [num_(kv_)heads * head_dim, hidden]; slice for this rank on dim 0.
        rank_slice = full_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        self.weight.data.narrow(0, offset, size).copy_(rank_slice)


class TPMHA(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        head_dim: int,
        tp_size: int,
        tp_rank: int,
        rope_base: float = 10000.0,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        assert n_heads % tp_size == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_heads_per_rank = n_heads // tp_size
        self.rope_base = rope_base
        self.qkv_proj = QKVParallelLinear(
            hidden, head_dim, n_heads, n_heads, tp_size, tp_rank, group=group
        )
        self.o_proj = RowParallelLinear(
            n_heads * head_dim, hidden, tp_size, tp_rank, group=group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)  # [B, T, 3 * n_heads_per_rank * head_dim]
        local_q_size = self.n_heads_per_rank * self.head_dim
        q, k, v = qkv.split([local_q_size, local_q_size, local_q_size], dim=-1)
        q = q.view(B, T, self.n_heads_per_rank, self.head_dim)
        k = k.view(B, T, self.n_heads_per_rank, self.head_dim)
        v = v.view(B, T, self.n_heads_per_rank, self.head_dim)

        cos, sin = build_rope_cache(
            T, self.head_dim, base=self.rope_base, device=x.device, dtype=x.dtype
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads_per_rank * self.head_dim)
        return self.o_proj(out)
