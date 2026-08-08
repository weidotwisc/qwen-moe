"""Tests for ex01: ColumnParallelLinear and RowParallelLinear."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.linear import RefLinear
from bootcamp.tests.conftest import DTYPES, tol

# USE_REFERENCE=1 swaps in the reference solution — useful for verifying the
# reference or running downstream exercises before finishing this one.
if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex01_linear_tp.reference import ColumnParallelLinear, RowParallelLinear
else:
    from bootcamp.ex01_linear_tp.solution import ColumnParallelLinear, RowParallelLinear

# Sizes chosen so all of {1, 2, 4, 8} divide cleanly.
HIDDEN_IN, HIDDEN_OUT, SEQ = 128, 256, 8


def _column_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    # Same seed on every rank → identical reference weight, so each rank's
    # weight_loader can be tested by slicing that shared reference.
    torch.manual_seed(0)
    ref = RefLinear(HIDDEN_IN, HIDDEN_OUT, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(SEQ, HIDDEN_IN, device=device, dtype=dtype)

    layer = ColumnParallelLinear(HIDDEN_IN, HIDDEN_OUT, world_size, rank).to(
        device=device, dtype=dtype
    )
    layer.weight_loader(ref.proj.weight.data)

    y_shard = layer(x)
    expected_shard = HIDDEN_OUT // world_size
    assert y_shard.shape == (SEQ, expected_shard), (
        f"rank {rank}: expected shard shape {(SEQ, expected_shard)}, got {tuple(y_shard.shape)}"
    )

    # Gather shards on every rank to compare (all_gather → all ranks see full result)
    gathered = [torch.empty_like(y_shard) for _ in range(world_size)]
    dist.all_gather(gathered, y_shard.contiguous())
    y_full = torch.cat(gathered, dim=-1)

    y_ref = ref(x)
    torch.testing.assert_close(y_full, y_ref, **tol(dtype))


def _row_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    torch.manual_seed(0)
    ref = RefLinear(HIDDEN_IN, HIDDEN_OUT, bias=False).to(device=device, dtype=dtype)
    x_full = torch.randn(SEQ, HIDDEN_IN, device=device, dtype=dtype)

    layer = RowParallelLinear(HIDDEN_IN, HIDDEN_OUT, world_size, rank).to(
        device=device, dtype=dtype
    )
    layer.weight_loader(ref.proj.weight.data)

    # Each rank feeds its input shard
    shard_size = HIDDEN_IN // world_size
    x_shard = x_full[:, rank * shard_size : (rank + 1) * shard_size].contiguous()

    y = layer(x_shard)
    assert y.shape == (SEQ, HIDDEN_OUT), (
        f"rank {rank}: expected {(SEQ, HIDDEN_OUT)}, got {tuple(y.shape)}"
    )

    y_ref = ref(x_full)
    torch.testing.assert_close(y, y_ref, **tol(dtype))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_column_parallel_linear(tp_size: int, dtype_str: str) -> None:
    require_gpus(tp_size)
    run_on_ranks(tp_size, _column_worker, dtype_str)


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_row_parallel_linear(tp_size: int, dtype_str: str) -> None:
    require_gpus(tp_size)
    run_on_ranks(tp_size, _row_worker, dtype_str)
