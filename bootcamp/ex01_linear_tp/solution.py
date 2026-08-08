"""Exercise 1 — Linear Tensor Parallelism.

Fill in ColumnParallelLinear and RowParallelLinear. Signatures match
nanovllm-jun/nanovllm/layers/linear.py so your solution can drop into nanovllm
directly once the whole course is done.

Rules:
    - Do NOT read nanovllm-jun/nanovllm/layers/linear.py while implementing.
    - Only torch.distributed.all_reduce (and optionally all_gather) are needed.
    - Bias is always False in these exercises — no bias TP trick.

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest -x bootcamp/tests/test_ex01_linear_tp.py -v
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class ColumnParallelLinear(nn.Module):
    """Shards a linear layer's OUTPUT dimension across `tp_size` ranks.

    Full weight is [out_features, in_features]. Each rank holds a
    [out_features // tp_size, in_features] slice. Forward returns the
    partial output on the sharded dim — NO collective. Downstream code
    (e.g. a RowParallelLinear, or an explicit all_gather) is responsible
    for either consuming the shard or gathering it.

    Args:
        in_features: total input dim.
        out_features: total output dim; must be divisible by tp_size.
        tp_size: number of TP ranks.
        tp_rank: this rank's id in [0, tp_size).
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        # TODO(you):
        # 1. assert out_features % tp_size == 0
        # 2. self.in_features / self.out_features / self.tp_size / self.tp_rank
        # 3. self.weight = nn.Parameter of shape (out_features // tp_size, in_features)
        raise NotImplementedError

    def weight_loader(self, full_weight: torch.Tensor) -> None:
        """Copy this rank's slice from `full_weight` [out_features, in_features]
        into self.weight. Use dim=0 (the out-features dim)."""
        # TODO(you): pick the (rank * shard_size)..(rank * shard_size + shard_size)
        # slice of full_weight and copy it into self.weight.data.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]. Returns [..., out_features // tp_size].
        # TODO(you): a single F.linear call. Do NOT all-reduce.
        raise NotImplementedError


class RowParallelLinear(nn.Module):
    """Shards a linear layer's INPUT dimension across `tp_size` ranks.

    Full weight is [out_features, in_features]. Each rank holds a
    [out_features, in_features // tp_size] slice. Forward expects x to be
    already sharded on the in-features dim (i.e. x.shape[-1] ==
    in_features // tp_size), computes a partial output, and all-reduces
    to produce a full-size replicated output.

    Args:
        in_features: total input dim; must be divisible by tp_size.
        out_features: total output dim (not sharded).
        tp_size: number of TP ranks.
        tp_rank: this rank's id.
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        # TODO(you):
        # 1. assert in_features % tp_size == 0
        # 2. store dims and rank info
        # 3. self.weight of shape (out_features, in_features // tp_size)
        raise NotImplementedError

    def weight_loader(self, full_weight: torch.Tensor) -> None:
        """Copy this rank's slice from `full_weight` [out_features, in_features]
        into self.weight. Use dim=1 (the in-features dim)."""
        # TODO(you)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features // tp_size]. Returns [..., out_features] (replicated).
        # TODO(you):
        # 1. F.linear on the local shard
        # 2. dist.all_reduce (SUM) to combine partial products across ranks
        raise NotImplementedError
