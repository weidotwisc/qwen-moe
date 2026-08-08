"""Reference solutions for ex01."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        assert out_features % tp_size == 0, (
            f"out_features={out_features} must be divisible by tp_size={tp_size}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.shard_size = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.shard_size, in_features))

    def weight_loader(self, full_weight: torch.Tensor) -> None:
        # full_weight is [out_features, in_features]; take rows for this rank.
        start = self.tp_rank * self.shard_size
        self.weight.data.copy_(full_weight[start : start + self.shard_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        assert in_features % tp_size == 0, (
            f"in_features={in_features} must be divisible by tp_size={tp_size}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.shard_size = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.shard_size))

    def weight_loader(self, full_weight: torch.Tensor) -> None:
        # full_weight is [out_features, in_features]; take columns for this rank.
        start = self.tp_rank * self.shard_size
        self.weight.data.copy_(full_weight[:, start : start + self.shard_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
