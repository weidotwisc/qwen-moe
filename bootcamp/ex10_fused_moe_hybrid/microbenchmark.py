"""Ex07 (Python expert loop) vs Ex10 (Ex09 Triton fused kernel) — scale sweep.

Parameterized by env vars (TP_SIZE, DP_SIZE, EP_SIZE) so the same script covers
single-node 4-GPU, 8-GPU, and 2-node 16-GPU under Ex08's TCP-over-net1-0
transport. Rank plumbing:

    world_size == TP_SIZE * DP_SIZE == EP_SIZE

    TP groups: contiguous rank blocks of TP_SIZE ranks each. On 16-GPU
    (TP=8, DP=2), tp_group_0 = {0..7} (lsf00) and tp_group_1 = {8..15}
    (lsf01) — TP is intra-node, so its all-reduce hits NVLink not TCP.
    EP group = whole world.

Uses Qwen3-30B-A3B dims (H=2048, I=768, E=128, top_k=8, bf16).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Callable

import torch
import torch.distributed as dist


HIDDEN = 2048
INTERMEDIATE = 768
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = 128
NUM_EXPERTS = 128
TOP_K = 8
NORM_TOPK_PROB = True

TP_SIZE = int(os.environ["TP_SIZE"])
DP_SIZE = int(os.environ["DP_SIZE"])
EP_SIZE = int(os.environ["EP_SIZE"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])

assert TP_SIZE * DP_SIZE == WORLD_SIZE, f"TP*DP={TP_SIZE*DP_SIZE} != WORLD_SIZE={WORLD_SIZE}"
assert EP_SIZE == WORLD_SIZE, f"EP={EP_SIZE} != WORLD_SIZE={WORLD_SIZE}"

# batch × seq shapes — same set used across scales for apples-to-apples.
CONFIGS = [
    (16, 512),
    (32, 1024),
    (64, 2048),
]

WARMUP = 3
TRIALS = 15
DTYPE = torch.bfloat16


def _time_block(build_layer: Callable, batch: int, seq: int, dp_start: int, dp_end: int, device: str) -> list[float]:
    layer = build_layer()
    torch.manual_seed(1337)
    x_full = torch.randn(batch, seq, HIDDEN, device=device, dtype=DTYPE)
    local_x = x_full[dp_start:dp_end]

    for _ in range(WARMUP):
        _ = layer(local_x)
    torch.cuda.synchronize()
    dist.barrier()

    times_ms: list[float] = []
    for _ in range(TRIALS):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = layer(local_x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    del layer
    torch.cuda.empty_cache()
    return times_ms


def _reduce_max(times_ms: list[float]) -> list[float]:
    t = torch.tensor(times_ms, dtype=torch.float64, device=f"cuda:{LOCAL_RANK}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def main() -> None:
    torch.cuda.set_device(LOCAL_RANK)
    device = f"cuda:{LOCAL_RANK}"
    dist.init_process_group("nccl", device_id=torch.device(device))

    # TP groups: contiguous blocks of TP_SIZE.
    tp_groups = []
    for g in range(DP_SIZE):
        ranks = list(range(g * TP_SIZE, (g + 1) * TP_SIZE))
        tp_groups.append(dist.new_group(ranks=ranks))
    tp_group = tp_groups[RANK // TP_SIZE]
    tp_rank = RANK % TP_SIZE
    dp_rank = RANK // TP_SIZE
    ep_group = None  # world
    ep_rank = RANK

    from bootcamp.ref.block import RefBlock
    from bootcamp.ex07_tp_ep_hybrid.reference import HybridBlock as Ex07Block
    from bootcamp.ex10_fused_moe_hybrid.reference import FusedHybridBlock as Ex10Block

    # Shared reference — same seed everywhere so weights match across ranks.
    torch.manual_seed(42)
    ref = RefBlock(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K, norm_topk_prob=NORM_TOPK_PROB,
    ).to(device=device, dtype=DTYPE)

    def _build(BlockCls):
        def _f():
            layer = BlockCls(
                hidden=HIDDEN, intermediate=INTERMEDIATE,
                n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
                num_experts=NUM_EXPERTS, top_k=TOP_K,
                tp_size=TP_SIZE, tp_rank=tp_rank, tp_group=tp_group,
                ep_size=EP_SIZE, ep_rank=ep_rank, ep_group=ep_group,
                norm_topk_prob=NORM_TOPK_PROB,
            ).to(device=device, dtype=DTYPE)
            expert_gate = [ref.moe.experts[e].gate_proj.weight.data for e in range(NUM_EXPERTS)]
            expert_up   = [ref.moe.experts[e].up_proj.weight.data   for e in range(NUM_EXPERTS)]
            expert_down = [ref.moe.experts[e].down_proj.weight.data for e in range(NUM_EXPERTS)]
            layer.weight_loader(
                attn_norm_weight=ref.attn_norm.weight.data,
                q_weight=ref.attn.q_proj.weight.data,
                k_weight=ref.attn.k_proj.weight.data,
                v_weight=ref.attn.v_proj.weight.data,
                o_weight=ref.attn.o_proj.weight.data,
                moe_norm_weight=ref.moe_norm.weight.data,
                gate_weight=ref.moe.gate.weight.data,
                expert_gate_weights=expert_gate,
                expert_up_weights=expert_up,
                expert_down_weights=expert_down,
            )
            return layer
        return _f

    results = []
    if RANK == 0:
        print(f"\n============================================================", flush=True)
        print(f"  Ex07 vs Ex10 microbench   TP={TP_SIZE}  DP={DP_SIZE}  EP={EP_SIZE}  world={WORLD_SIZE}", flush=True)
        print(f"  H={HIDDEN} I={INTERMEDIATE} E={NUM_EXPERTS} top_k={TOP_K}  dtype={DTYPE}", flush=True)
        print(f"  warmup={WARMUP} trials={TRIALS}", flush=True)
        print(f"============================================================", flush=True)

    for batch, seq in CONFIGS:
        if batch < DP_SIZE:
            continue
        assert batch % DP_SIZE == 0
        local_B = batch // DP_SIZE
        dp_start = dp_rank * local_B
        dp_end = dp_start + local_B
        if (local_B * seq) % TP_SIZE != 0:
            continue

        if RANK == 0:
            print(f"\n  --- batch={batch} seq={seq}  (local_B={local_B}, N_tp={local_B*seq}) ---", flush=True)

        ex07_times = _time_block(_build(Ex07Block), batch, seq, dp_start, dp_end, device)
        ex10_times = _time_block(_build(Ex10Block), batch, seq, dp_start, dp_end, device)

        ex07_wall = _reduce_max(ex07_times)
        ex10_wall = _reduce_max(ex10_times)

        if RANK == 0:
            e07_med = statistics.median(ex07_wall)
            e10_med = statistics.median(ex10_wall)
            e07_std = statistics.stdev(ex07_wall) if len(ex07_wall) > 1 else 0.0
            e10_std = statistics.stdev(ex10_wall) if len(ex10_wall) > 1 else 0.0
            speedup = e07_med / e10_med
            results.append({
                "world": WORLD_SIZE, "tp": TP_SIZE, "dp": DP_SIZE, "ep": EP_SIZE,
                "batch": batch, "seq": seq,
                "ex07_ms": e07_med, "ex07_std": e07_std,
                "ex10_ms": e10_med, "ex10_std": e10_std,
                "speedup": speedup,
            })
            print(
                f"  ex07 (loop):  {e07_med:7.2f} ± {e07_std:5.2f} ms   "
                f"ex10 (fused): {e10_med:7.2f} ± {e10_std:5.2f} ms   "
                f"speedup: {speedup:4.2f}x",
                flush=True,
            )

    if RANK == 0:
        out_dir = os.environ.get("BENCH_OUT_DIR", "/tmp")
        out_path = f"{out_dir}/ex10_bench_world{WORLD_SIZE}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  wrote {out_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
