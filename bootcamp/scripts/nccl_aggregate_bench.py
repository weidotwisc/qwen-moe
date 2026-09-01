"""Aggregate cross-node NCCL bandwidth: N concurrent 2-rank allreduce groups.

Launch shape (torchrun): world_size = 2N, N ranks per node.
Subgroup layout: for i in 0..N-1, group_i pairs rank i (lsf00) with rank N+i (lsf01).
All N groups run allreduce concurrently, so we exercise up to N cross-node pairs
at once. If NCCL/GDR maps different GPU-index ranks to different HCAs via NUMA
affinity, aggregate bandwidth = sum of per-group bandwidths ≥ single-pair.

Reports each pair's bandwidth AND the aggregate sum.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


SIZES_BYTES = [
    16 * 1024 * 1024,   #  16 MiB
    128 * 1024 * 1024,  # 128 MiB
    512 * 1024 * 1024,  # 512 MiB
]


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size // 2))
    n_groups = ranks_per_node
    assert world_size == 2 * n_groups, f"expect world_size = 2 * ranks_per_node, got ws={world_size}, rpn={ranks_per_node}"

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    # Build N pair-groups. Every rank must enter every new_group call (collective).
    subgroups = []
    for i in range(n_groups):
        g = dist.new_group([i, n_groups + i])
        subgroups.append(g)

    # Which subgroup do I belong to?
    my_gid = rank if rank < n_groups else rank - n_groups
    my_group = subgroups[my_gid]

    if rank == 0:
        print(f"\n=== Aggregate cross-node bench   world_size={world_size}   "
              f"concurrent_pairs={n_groups}   local_rank->group  gpu:{local_rank}->g{my_gid} ===",
              flush=True)

    for nbytes in SIZES_BYTES:
        nfloat = nbytes // 4
        t = torch.ones(nfloat, dtype=torch.float32, device=f"cuda:{local_rank}")

        # Warmup within my subgroup
        for _ in range(3):
            dist.all_reduce(t, group=my_group)
        torch.cuda.synchronize()

        # Global barrier so all pairs start together — this is what makes the
        # measurement about aggregate concurrent throughput.
        dist.barrier()
        torch.cuda.synchronize()

        iters = max(5, min(50, (100 * 1024 * 1024) // nbytes))
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t, group=my_group)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        my_algbw = (nbytes * iters) / elapsed / 1e9  # 2-rank ring: algbw == busbw

        # Gather per-pair bandwidths to rank 0 in the WORLD group.
        gathered = [torch.zeros(1, device=f"cuda:{local_rank}") for _ in range(world_size)] if rank == 0 else None
        payload = torch.tensor([my_algbw], device=f"cuda:{local_rank}")
        dist.gather(payload, gathered, dst=0)

        if rank == 0:
            per_pair = [gathered[i].item() for i in range(n_groups)]  # only need lsf00 side; lsf01 side reports the same pair
            total = sum(per_pair)
            per_pair_str = "  ".join(f"g{i}={v:5.2f}" for i, v in enumerate(per_pair))
            print(f"  {nbytes:>12,d}  iters={iters:>3d}  per-pair GB/s: [{per_pair_str}]  "
                  f"AGGREGATE={total:6.2f} GB/s",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
