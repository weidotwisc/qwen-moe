"""Tests for ex00: the five torch.distributed wrappers.

Each test spawns N ranks, has each rank contribute a distinctive payload,
and verifies the collective's post-condition holds byte-exactly on every rank.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bootcamp.dist_utils import require_gpus, run_on_ranks

# Set USE_REFERENCE=1 to run tests against the reference solution instead of
# your own. Useful for verifying the reference itself works, or for running
# downstream exercises before you've finished this one.
if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex00_dist_primer.reference import (  # noqa: F401
        all_gather_into_tensor_wrapper,
        all_reduce_sum,
        all_to_all_equal,
        all_to_all_variable,
        destroy_dist,
        init_dist,
        reduce_scatter_sum_tensor_wrapper,
    )
else:
    from bootcamp.ex00_dist_primer.solution import (  # noqa: F401
        all_gather_into_tensor_wrapper,
        all_reduce_sum,
        all_to_all_equal,
        all_to_all_variable,
        destroy_dist,
        init_dist,
        reduce_scatter_sum_tensor_wrapper,
    )


# ---------- init / destroy ----------
# These tests do NOT use `run_on_ranks` — that helper uses the canonical
# `bootcamp.dist_utils` init.  Here we want to exercise the STUDENT's
# init/destroy directly, so we spawn workers with `mp.spawn` and let them
# call init_dist / destroy_dist themselves.

_INIT_TEST_PORT_BASE = 29100  # separate range from dist_utils.pick_port to avoid collision


def _init_and_destroy_worker(rank: int, world_size: int, port: int) -> None:
    init_dist(rank, world_size, port)
    try:
        # Post-condition assertions from the docstring.
        assert dist.is_initialized(), "init_dist did not initialize the process group"
        assert dist.get_rank() == rank, f"expected rank {rank}, got {dist.get_rank()}"
        assert dist.get_world_size() == world_size, (
            f"expected world_size {world_size}, got {dist.get_world_size()}"
        )
        assert torch.cuda.current_device() == rank, (
            f"expected device {rank}, got {torch.cuda.current_device()}"
        )

        # Smoke: actually do a collective. If init was subtly broken (e.g. wrong
        # backend, mismatched world_size), this hangs or errors.
        x = torch.tensor([float(rank)], device=f"cuda:{rank}")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        expected = float(world_size * (world_size - 1) // 2)
        assert x.item() == expected, f"all_reduce SUM: expected {expected}, got {x.item()}"
    finally:
        destroy_dist()
        assert not dist.is_initialized(), "destroy_dist did not tear down the process group"


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_init_and_destroy(world_size: int) -> None:
    require_gpus(world_size)
    port = _INIT_TEST_PORT_BASE + world_size  # deterministic per size, avoids collision
    mp.spawn(  # type: ignore[attr-defined]
        _init_and_destroy_worker, args=(world_size, port), nprocs=world_size, join=True
    )


# ---------- all_reduce ----------

def _all_reduce_worker(rank: int, world_size: int) -> None:
    device = f"cuda:{rank}"
    x = torch.tensor([float(rank)], device=device)
    all_reduce_sum(x)
    expected = torch.tensor([float(world_size * (world_size - 1) // 2)], device=device)
    torch.testing.assert_close(x, expected)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_all_reduce_sum(world_size: int) -> None:
    require_gpus(world_size)
    run_on_ranks(world_size, _all_reduce_worker)


# ---------- all_gather ----------

def _all_gather_worker(rank: int, world_size: int) -> None:
    device = f"cuda:{rank}"
    K = 3
    x = torch.full((K,), float(rank), device=device)
    out = all_gather_into_tensor_wrapper(x)
    # Post: out has shape (world_size, K); out[r] == rank r's x.
    assert out.shape == (world_size, K), f"got {tuple(out.shape)}"
    expected = torch.arange(world_size, dtype=torch.float32, device=device).unsqueeze(-1).expand(world_size, K)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_all_gather_into_tensor(world_size: int) -> None:
    require_gpus(world_size)
    run_on_ranks(world_size, _all_gather_worker)


# ---------- reduce_scatter ----------

def _reduce_scatter_worker(rank: int, world_size: int) -> None:
    device = f"cuda:{rank}"
    K = 3
    # Each rank contributes an (world_size, K) tensor where row j = tensor([rank + j*10]) * ones(K).
    # After reduce_scatter with SUM, rank j gets the sum over ranks of row j:
    #   sum_r (r + j*10) * ones(K) = (world_size*(world_size-1)/2 + world_size*j*10) * ones(K)
    x = torch.stack([torch.full((K,), float(rank + j * 10), device=device) for j in range(world_size)])
    out = reduce_scatter_sum_tensor_wrapper(x)
    assert out.shape == (K,), f"got {tuple(out.shape)}"
    expected_val = float(world_size * (world_size - 1) // 2 + world_size * rank * 10)
    expected = torch.full((K,), expected_val, device=device)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_reduce_scatter_sum(world_size: int) -> None:
    require_gpus(world_size)
    run_on_ranks(world_size, _reduce_scatter_worker)


# ---------- all_to_all equal ----------

def _all_to_all_equal_worker(rank: int, world_size: int) -> None:
    device = f"cuda:{rank}"
    # rank i sends chunk[j] = rank*100 + j to rank j; K=1 element per chunk.
    x = torch.tensor([rank * 100 + j for j in range(world_size)], dtype=torch.float32, device=device)
    out = all_to_all_equal(x)
    # Post on rank j: out[i] == rank i's chunk to j == i*100 + j.
    expected = torch.tensor([i * 100 + rank for i in range(world_size)], dtype=torch.float32, device=device)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_all_to_all_equal(world_size: int) -> None:
    require_gpus(world_size)
    run_on_ranks(world_size, _all_to_all_equal_worker)


# ---------- all_to_all variable  (THE EP dispatch primitive) ----------

def _all_to_all_variable_worker(rank: int, world_size: int) -> None:
    """Scenario:
       * Every rank sends `j+1` elements to rank j  (in_splits = [1, 2, 3, ...]).
       * So every rank receives (rank+1) elements from each of world_size senders
         → out_splits[j] = rank+1 for all j.
       * Each rank fills its input with 1000*rank + local_idx so we can tell
         provenance in the output.
    """
    device = f"cuda:{rank}"
    in_splits = [j + 1 for j in range(world_size)]        # [1, 2, ..., W]
    out_splits = [rank + 1 for _ in range(world_size)]    # [rank+1] * W
    total_in = sum(in_splits)                             # W*(W+1)/2
    x = torch.arange(total_in, dtype=torch.float32, device=device) + 1000.0 * rank

    out = all_to_all_variable(x, in_splits, out_splits)
    assert out.shape == (sum(out_splits),), f"got {tuple(out.shape)}"

    # Post: on this rank, the block of size (rank+1) coming from sender i
    # equals sender i's slice for this rank.  Sender i's slice for this rank
    # starts at sum(in_splits[:rank]) = rank*(rank+1)//2 in the sender's local
    # input, has length rank+1, and its values are 1000*i + arange(...).
    slice_start = rank * (rank + 1) // 2
    slice_len = rank + 1
    expected_parts = []
    for i in range(world_size):
        expected_parts.append(
            torch.arange(slice_start, slice_start + slice_len, dtype=torch.float32, device=device) + 1000.0 * i
        )
    expected = torch.cat(expected_parts)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_all_to_all_variable(world_size: int) -> None:
    require_gpus(world_size)
    run_on_ranks(world_size, _all_to_all_variable_worker)
