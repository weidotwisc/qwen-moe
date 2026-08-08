"""Distributed helpers for the TP/EP bootcamp exercises.

Public API:
    run_on_ranks(nprocs, worker_fn, *args)
        Spawn `nprocs` processes, each running worker_fn(rank, world_size, *args).
        Handles NCCL init + teardown; propagates AssertionError from any rank.

    require_gpus(n)
        Skip the current pytest test if fewer than n CUDA devices are visible.

The workers themselves must not call init/destroy_process_group — that's done
around them by run_on_ranks. Just do their compute + asserts.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Bump the port for each run_on_ranks call so back-to-back tests don't collide
# on a TIME_WAIT-held socket.
_next_port = 29500


def _pick_port() -> int:
    global _next_port
    _next_port += 1
    return _next_port


def _init(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Torch >=2.2 renamed NCCL_ASYNC_ERROR_HANDLING -> TORCH_NCCL_ASYNC_ERROR_HANDLING;
    # old name still works but emits a deprecation warning.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _wrapped(rank: int, world_size: int, port: int, worker_fn, worker_args) -> None:
    _init(rank, world_size, port)
    try:
        worker_fn(rank, world_size, *worker_args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_on_ranks(nprocs: int, worker_fn, *args) -> None:
    """Spawn `nprocs` processes running worker_fn(rank, world_size, *args).

    Blocks until all workers exit. If any raises (including AssertionError from
    a test), torch.multiprocessing.spawn re-raises it in the parent as a
    ProcessRaisedException — pytest reports that as a failure.
    """
    port = _pick_port()
    mp.spawn(_wrapped, args=(nprocs, port, worker_fn, args), nprocs=nprocs, join=True)


def require_gpus(n: int) -> None:
    """Call from inside a pytest test to skip if fewer than n GPUs are visible."""
    have = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if have < n:
        pytest.skip(f"needs {n} GPUs, have {have}")
