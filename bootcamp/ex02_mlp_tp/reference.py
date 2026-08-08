"""Reference solutions for ex02.

Imports the ex01 REFERENCE (not solution) so this file works stand-alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bootcamp.ex01_linear_tp.reference import ColumnParallelLinear, RowParallelLinear


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, in_features: int, output_sizes: list[int], tp_size: int, tp_rank: int
    ) -> None:
        super().__init__(in_features, sum(output_sizes), tp_size, tp_rank)
        self.output_sizes = output_sizes

    def weight_loader(self, full_weight: torch.Tensor, shard_id: int) -> None:  # type: ignore[override]
        # Offset within self.weight (which is [sum(output_sizes)/tp_size, in_features]).
        shard_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        shard_size = self.output_sizes[shard_id] // self.tp_size
        # Slice of full_weight (which is [output_sizes[shard_id], in_features]) for this rank.
        rank_start = self.tp_rank * shard_size
        rank_slice = full_weight[rank_start : rank_start + shard_size]
        self.weight.data.narrow(0, shard_offset, shard_size).copy_(rank_slice)


class TPSwiGLUMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden, [intermediate, intermediate], tp_size, tp_rank
        )
        self.down_proj = RowParallelLinear(intermediate, hidden, tp_size, tp_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
