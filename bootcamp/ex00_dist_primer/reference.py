"""Reference solutions for ex00. Match the specs in solution.py exactly.

Look here if you're stuck on a wrapper; the point of the exercise is the
spec + one API call, not cleverness.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_dist(rank: int, world_size: int, port: int = 29500) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def destroy_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(x: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


def all_gather_into_tensor_wrapper(
    x: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    output = torch.empty((world_size, *x.shape), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(output, x, group=group)
    return output


def reduce_scatter_sum_tensor_wrapper(
    x: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    assert x.shape[0] == world_size, (
        f"reduce_scatter expects leading dim == world_size ({world_size}), got {x.shape[0]}"
    )
    output = torch.empty(x.shape[1:], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(output, x, op=dist.ReduceOp.SUM, group=group)
    return output


def all_to_all_equal(
    x: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=group)
    return output


def all_to_all_variable(
    x: torch.Tensor,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    total_output = sum(output_split_sizes)
    output = torch.empty((total_output, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        output,
        x,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output
