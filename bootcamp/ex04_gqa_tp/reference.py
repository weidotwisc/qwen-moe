"""Reference solution for ex04 — GQA under TP with KV-head replication."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex01_linear_tp.reference import ColumnParallelLinear, RowParallelLinear
from bootcamp.rope import apply_rope, build_rope_cache


class QKVParallelLinearGQA(ColumnParallelLinear):
    def __init__(
        self,
        hidden: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        assert num_heads % tp_size == 0, (
            f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
        )
        assert (num_kv_heads % tp_size == 0) or (tp_size % num_kv_heads == 0), (
            f"num_kv_heads ({num_kv_heads}) and tp_size ({tp_size}) must be "
            f"one divisible by the other (for kv sharding vs replication rule)"
        )

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_heads_per_rank = num_heads // tp_size
        self.num_kv_heads_per_rank = max(1, num_kv_heads // tp_size)
        self.num_kv_replicas = max(1, tp_size // num_kv_heads)
        self.q_size_per_rank = self.num_heads_per_rank * head_dim
        self.kv_size_per_rank = self.num_kv_heads_per_rank * head_dim

        # Total merged output for the parent — includes replicated KV storage.
        # Per-rank storage = q_size + 2 * kv_size; total = tp_size * per_rank.
        # Equivalently: (num_heads + 2 * num_kv_heads * num_kv_replicas) * head_dim.
        output_size = tp_size * (self.q_size_per_rank + 2 * self.kv_size_per_rank)
        super().__init__(hidden, output_size, tp_size, tp_rank, group=group)

    def weight_loader(self, full_weight: torch.Tensor, shard_id: str) -> None:  # type: ignore[override]
        if shard_id == "q":
            offset, length = 0, self.q_size_per_rank
            rank_slice = full_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        elif shard_id == "k":
            offset, length = self.q_size_per_rank, self.kv_size_per_rank
            rank_slice = self._kv_slice(full_weight)
        elif shard_id == "v":
            offset, length = self.q_size_per_rank + self.kv_size_per_rank, self.kv_size_per_rank
            rank_slice = self._kv_slice(full_weight)
        else:
            raise ValueError(f"shard_id must be 'q'|'k'|'v', got {shard_id!r}")

        self.weight.data.narrow(0, offset, length).copy_(rank_slice)

    def _kv_slice(self, full_kv_weight: torch.Tensor) -> torch.Tensor:
        """Get this rank's slice of K or V, handling both sharding and replication."""
        if self.num_kv_heads >= self.tp_size:
            # Normal sharding: chunk into tp_size, take this rank's chunk.
            return full_kv_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        else:
            # Replicated: chunk into num_kv_heads, take chunk `tp_rank // num_kv_replicas`.
            # Multiple ranks receive the same slice.
            kv_chunk_id = self.tp_rank // self.num_kv_replicas
            return full_kv_weight.chunk(self.num_kv_heads, dim=0)[kv_chunk_id]


class TPGQA(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        tp_size: int,
        tp_rank: int,
        rope_base: float = 10000.0,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        assert n_heads % tp_size == 0
        assert (n_kv_heads % tp_size == 0) or (tp_size % n_kv_heads == 0)

        self.hidden = hidden
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_heads_per_rank = n_heads // tp_size
        self.n_kv_heads_per_rank = max(1, n_kv_heads // tp_size)
        self.n_rep = self.n_heads_per_rank // self.n_kv_heads_per_rank
        self.rope_base = rope_base

        self.qkv_proj = QKVParallelLinearGQA(
            hidden, head_dim, n_heads, n_kv_heads, tp_size, tp_rank, group=group
        )
        self.o_proj = RowParallelLinear(
            n_heads * head_dim, hidden, tp_size, tp_rank, group=group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)  # [B, T, q_size + 2*kv_size]
        q_size = self.n_heads_per_rank * self.head_dim
        kv_size = self.n_kv_heads_per_rank * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.view(B, T, self.n_heads_per_rank, self.head_dim)
        k = k.view(B, T, self.n_kv_heads_per_rank, self.head_dim)
        v = v.view(B, T, self.n_kv_heads_per_rank, self.head_dim)

        cos, sin = build_rope_cache(
            T, self.head_dim, base=self.rope_base, device=x.device, dtype=x.dtype
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Broadcast KV heads up to Q's per-rank head count.
        k = k.repeat_interleave(self.n_rep, dim=2) # weiz: BTHD, we need to do hard bcast because SDPA requires dimension matching inputs
        v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads_per_rank * self.head_dim)
        return self.o_proj(out)
