"""Exercise 0 — torch.distributed primer.

Seven tasks. First two are the NCCL init/teardown boilerplate that every
distributed script needs (MPI_Init / MPI_Finalize equivalents). Remaining
five are typed wrappers around the collective primitives used throughout
the course.

Each function has a docstring pre/post condition that reads like a formal
spec. The paper's verification story treats them as opaque abstract
operations with these specs; keep the specs true and everything downstream
stays sound.

MPI mental model:
    init_dist(rank, world_size, port)  ↔  MPI_Init  (roughly; plus device binding)
    destroy_dist()                     ↔  MPI_Finalize
    all_reduce_sum(x)                  ↔  MPI_Allreduce(SUM)
    all_gather_into_tensor_wrapper(x)  ↔  MPI_Allgather
    reduce_scatter_tensor_wrapper(x)   ↔  MPI_Reduce_scatter_block  (SUM)
    all_to_all_equal(x)                ↔  MPI_Alltoall
    all_to_all_variable(x, is, os)     ↔  MPI_Alltoallv
"""

from __future__ import annotations

import os

import torch
from torch.cpu import is_initialized
import torch.distributed as dist


# ============================================================================
# Task 1: init_dist
# ============================================================================

def init_dist(rank: int, world_size: int, port: int = 29500) -> None:
    """Initialize the NCCL process group on this rank.

    Pre:   `torch.cuda.device_count() > rank`.
           No process group is currently initialized on this process.
    Post:  `dist.is_initialized()` is True.
           `dist.get_rank() == rank`.
           `dist.get_world_size() == world_size`.
           `torch.cuda.current_device() == rank`.

    Ordering matters: `torch.cuda.set_device(rank)` MUST happen BEFORE
    `dist.init_process_group("nccl", ...)`, because NCCL binds to whichever
    device is current at init time. If two ranks are current on the same
    device (e.g., you forgot set_device), NCCL will silently deadlock or
    corrupt buffers, not error.

    The port is passed in (rather than hard-coded) because tests re-init many
    times back-to-back and would otherwise collide on a TIME_WAIT-held socket.
    """
    # TODO(you):
    # 1. os.environ["MASTER_ADDR"] = "127.0.0.1"
    # 2. os.environ["MASTER_PORT"] = str(port)
    # 3. os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    #      Makes NCCL raise on the offending rank instead of hanging when a
    #      collective fails mid-run.  Torch >=2.2 renamed this from the older
    #      NCCL_ASYNC_ERROR_HANDLING (which still works but warns).
    # 4. torch.cuda.set_device(rank)
    # 5. dist.init_process_group("nccl", rank=rank, world_size=world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    gpu_count = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpu_count)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



# ============================================================================
# Task 2: destroy_dist
# ============================================================================

def destroy_dist() -> None:
    """Tear down the process group.

    Pre:   None (idempotent — safe to call when no group is initialized).
    Post:  `dist.is_initialized()` is False.
    """
    # TODO(you): `dist.destroy_process_group()` if `dist.is_initialized()`.
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Tasks 3-7: Collective wrappers
# ============================================================================


def all_reduce_sum(x: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """SUM-reduce `x` across `group`; every rank ends up with the sum.

    Pre:   x has the SAME shape and dtype on every rank of `group`.
           x is on the CUDA device bound to this rank.
    Post:  x is mutated in place to be sum_{r in group} x_r.
           Return the same tensor for chaining.
    """
    # TODO(you): one call to dist.all_reduce with op=SUM and the given group.
    raise NotImplementedError


def all_gather_into_tensor_wrapper(
    x: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Gather `x` from every rank into a single output on every rank.

    Uses `dist.all_gather_into_tensor` (the modern single-buffer variant),
    not the legacy `dist.all_gather` which takes a list of tensors.

    Pre:   x has SAME shape and dtype on every rank; world_size == len(group).
    Post:  return a NEW tensor of shape (world_size,) + x.shape,
           contiguous, with output[r] == x_r for each rank r in `group`.
    """
    # TODO(you):
    # 1. Compute world_size (dist.get_world_size(group=group)).
    # 2. Allocate output of shape (world_size, *x.shape) on the same device/dtype.
    # 3. dist.all_gather_into_tensor(output, x, group=group).
    # 4. Return output.
    raise NotImplementedError


def reduce_scatter_sum_tensor_wrapper(
    x: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """SUM-reduce along axis 0, then scatter shards across `group`.

    Inverse of all_gather + sum: instead of everyone getting the total,
    everyone gets 1/world_size-th of the total.

    Pre:   x has shape (world_size, K...) with identical shape and dtype on every rank.
    Post:  return a NEW tensor of shape (K...) equal to
           sum_{r in group} x_r[this_rank]  on this rank.
    """
    # TODO(you):
    # 1. world_size = dist.get_world_size(group=group).
    # 2. assert x.shape[0] == world_size.
    # 3. Allocate output of shape x.shape[1:].
    # 4. dist.reduce_scatter_tensor(output, x, op=SUM, group=group).
    # 5. Return output.
    raise NotImplementedError


def all_to_all_equal(
    x: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Equal-splits all-to-all along axis 0 (the "transpose" pattern).

    Pre:   x.shape[0] % world_size == 0.
    Post:  return a NEW tensor of the same shape as x, such that on rank r
             output[i*K : (i+1)*K] == input_r_from_rank_i,
           where K = x.shape[0] // world_size.
    """
    # TODO(you):
    # 1. Allocate `output` of the same shape/dtype/device as x.
    # 2. dist.all_to_all_single(output, x, group=group).
    # 3. Return output.
    raise NotImplementedError


def all_to_all_variable(
    x: torch.Tensor,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Variable-splits all-to-all — the EP dispatch primitive.

    Pre:   len(input_split_sizes) == len(output_split_sizes) == world_size.
           sum(input_split_sizes) == x.shape[0].
           For every rank r, input_split_sizes[r] on this rank
              == output_split_sizes[this_rank] on rank r.
           (This precondition is the load-bearing one — if it doesn't hold,
           NCCL will deadlock or produce nonsense. In practice the two
           splits vectors are negotiated with an `all_gather_into_tensor` of
           `input_split_sizes` beforehand.)
    Post:  return a NEW tensor of shape (sum(output_split_sizes), *x.shape[1:]).
           On this rank, output[sum(output_split_sizes[:i]) : sum(output_split_sizes[:i+1])]
             == the i-th slice of x on rank i (where "i-th slice" is
             defined by input_split_sizes on rank i).
    """
    # TODO(you):
    # 1. Compute total_output = sum(output_split_sizes).
    # 2. Allocate `output` of shape (total_output, *x.shape[1:]).
    # 3. dist.all_to_all_single(output, x,
    #        output_split_sizes=output_split_sizes,
    #        input_split_sizes=input_split_sizes,
    #        group=group).
    # 4. Return output.
    raise NotImplementedError
