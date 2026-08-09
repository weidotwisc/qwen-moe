"""Exercise 3 — Multi-Head Attention under TP.

Fill in:

1. `QKVParallelLinear` — a fused Q+K+V projection sharded on the head axis.
   Under TP-N with `num_heads` heads, each rank owns `num_heads / N` heads.
   Because Q, K, V all shard the same way for standard MHA, we can pack them
   into a single [3 * (num_heads / N) * head_dim, hidden] weight per rank
   and issue one GEMM.

   This exercise ASSUMES num_heads == num_kv_heads (pure MHA). Exercise 4
   extends this to GQA where num_kv_heads < num_heads, with KV-head
   replication when tp_size > num_kv_heads.

2. `TPMHA` — glue: QKVParallelLinear → RoPE on the local head shard → SDPA
   → RowParallelLinear for o_proj.

Depends on your ex01 solution.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex01_linear_tp.solution import ColumnParallelLinear, RowParallelLinear
from bootcamp.rope import apply_rope, build_rope_cache


class QKVParallelLinear(ColumnParallelLinear):
    """Packed Q/K/V column-parallel linear layer.

    Full "logical" weight layout on the output dim, in order:
        [ Q_head0 ... Q_head_{H-1}
          K_head0 ... K_head_{H-1}
          V_head0 ... V_head_{H-1} ]
    where each head contributes `head_dim` rows.

    Under TP-N, each rank owns heads [rank * H/N, (rank+1) * H/N) of Q, K, and V.
    Total rows per rank: 3 * (H/N) * head_dim.

    Weight is loaded per projection with a shard_id in {"q", "k", "v"}. The
    full projection weight passed in has shape [H * head_dim, hidden].
    """

    def __init__(
        self,
        hidden: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        assert num_heads == num_kv_heads, (
            "ex03 handles MHA only (num_heads == num_kv_heads). "
            "GQA is exercise 4."
        )
        assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size_per_rank = (num_heads // tp_size) * head_size
        self.kv_size_per_rank = (num_kv_heads // tp_size) * head_size

        # Total merged output = Q_all + K_all + V_all.
        output_size = (num_heads + 2 * num_kv_heads) * head_size
        super().__init__(hidden, output_size, tp_size, tp_rank, group=group)

    def weight_loader(self, full_weight: torch.Tensor, shard_id: str) -> None:  # type: ignore[override]
        """Copy this rank's slice of a full [num_(kv_)heads * head_size, hidden]
        projection weight into the appropriate section of self.weight.

        shard_id: "q", "k", or "v".
        """
        # TODO(you):
        # 1. Figure out where in self.weight (dim 0) this shard_id begins:
        #    - "q" starts at 0
        #    - "k" starts at q_size_per_rank
        #    - "v" starts at q_size_per_rank + kv_size_per_rank
        #    Its length is q_size_per_rank (for "q") or kv_size_per_rank (for "k"/"v").
        # 2. Split full_weight into tp_size chunks on dim 0 and take this rank's.
        # 3. Copy that chunk into self.weight.data.narrow(0, offset, length).
        raise NotImplementedError


class TPMHA(nn.Module):
    """Full multi-head attention block with TP.

    Forward: x → QKVParallelLinear → RoPE(Q,K) → SDPA(causal) → RowParallelLinear.
    Every rank sees all sequence tokens; heads are sharded.
    """

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

        # TODO(you):
        # 1. self.qkv_proj = QKVParallelLinear(hidden, head_dim, n_heads, n_heads, tp_size, tp_rank, group=group)
        # 2. self.o_proj = RowParallelLinear(n_heads * head_dim, hidden, tp_size, tp_rank, group=group)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden]
        B, T, _ = x.shape
        # TODO(you):
        # 1. qkv = self.qkv_proj(x)   # [B, T, 3 * n_heads_per_rank * head_dim]
        # 2. Split into q, k, v — each [B, T, n_heads_per_rank * head_dim].
        #    Careful: they are NOT three equal thirds of the last dim in the
        #    packed layout; use `.split(...)` with an explicit list of sizes.
        # 3. View each as [B, T, n_heads_per_rank, head_dim].
        # 4. Build RoPE cos/sin cache for head_dim and this seq length.
        # 5. Apply RoPE to q and k (not v).
        # 6. Transpose to [B, n_heads_per_rank, T, head_dim] for SDPA.
        # 7. F.scaled_dot_product_attention(q, k, v, is_causal=True).
        # 8. Transpose back to [B, T, n_heads_per_rank, head_dim], reshape to
        #    [B, T, n_heads_per_rank * head_dim] — this is the sharded input
        #    for the RowParallelLinear.
        # 9. return self.o_proj(...)
        raise NotImplementedError
