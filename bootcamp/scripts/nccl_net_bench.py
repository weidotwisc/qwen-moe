"""NCCL allreduce bandwidth benchmark, torchrun-compatible.

Runs identically for single-node (torchrun --standalone) and multi-node
(torchrun --nnodes=N --node_rank=k --master_addr=... --master_port=...). Which
NCCL transport actually runs — NVLink, RoCE/IB, or TCP — is determined by env
vars set BEFORE this process starts. See nccl_net_bench.sh (single-node) and
nccl_net_bench_2node.sh (2-node) for the switching wrappers.

Rank/world_size come from torchrun env (RANK, LOCAL_RANK, WORLD_SIZE). NET_MODE
is read for output labeling only.

algbw = bytes_per_op * iters / elapsed   (raw data volume moved / second)
busbw = algbw * 2*(N-1)/N                (ring-allreduce bus bandwidth)
For 2 ranks: busbw == algbw. For 16 ranks: busbw ≈ 1.875 × algbw.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


SIZES_BYTES = [
    64 * 1024,          #  64 KiB
    1 * 1024 * 1024,    #   1 MiB
    16 * 1024 * 1024,   #  16 MiB
    128 * 1024 * 1024,  # 128 MiB
    512 * 1024 * 1024,  # 512 MiB
    1024 * 1024 * 1024, #   1 GiB
]


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    mode = os.environ.get("NET_MODE", "unset")
    if rank == 0:
        node_count = int(os.environ.get("GROUP_WORLD_SIZE", world_size)) // int(
            os.environ.get("LOCAL_WORLD_SIZE", world_size)
        )
        print(
            f"\n=== NET_MODE={mode}  world_size={world_size}  "
            f"nodes~={node_count}  local_size={os.environ.get('LOCAL_WORLD_SIZE', '?')} ===",
            flush=True,
        )

    for nbytes in SIZES_BYTES:
        nfloat = nbytes // 4
        t = torch.ones(nfloat, dtype=torch.float32, device=f"cuda:{local_rank}")

        for _ in range(5):
            dist.all_reduce(t)
        torch.cuda.synchronize()

        iters = max(10, min(200, (200 * 1024 * 1024) // nbytes))
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        algbw = (nbytes * iters) / elapsed / 1e9
        busbw = algbw * 2 * (world_size - 1) / world_size

        if rank == 0:
            per_op_ms = elapsed * 1000 / iters
            print(
                f"  {nbytes:>12,d}  iters={iters:>4d}  {per_op_ms:>8.3f} ms/op  "
                f"algbw={algbw:>7.2f} GB/s  busbw={busbw:>7.2f} GB/s",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
