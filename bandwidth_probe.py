"""Quick 2-rank NCCL bandwidth probe: rank 0 on one node, rank 1 on the other."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"world_size={world_size}, dtype=bf16")
        print(f"{'size (MB)':>12} {'time (ms)':>12} {'bw (GB/s)':>12}")
        print("-" * 40)

    sizes_mb = [1, 10, 50, 100, 500, 1000]
    for sz_mb in sizes_mb:
        numel = sz_mb * 1024 * 1024 // 2  # bf16 = 2 bytes/elem
        x = torch.ones(numel, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()

        # Time
        n = 10
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_s = (t1 - t0) / n
        # Ring all_reduce per-rank bandwidth: 2*(N-1)/N * tensor_bytes
        bytes_transferred = 2 * (world_size - 1) / world_size * sz_mb * 1024 * 1024
        gbps = bytes_transferred / avg_s / 1e9

        if rank == 0:
            print(f"{sz_mb:>12} {avg_s * 1000:>12.2f} {gbps:>12.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
