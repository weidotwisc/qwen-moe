"""Lean (single all_reduce) vs Dispatch (all_to_all × 2), both variants at
TP=EP=world_size with DP=1 (replicated input across the EP group). Works
single-node (world=8) or cross-node (world=16 via TCP over net1-0).

Compares four blocks side-by-side at batch=16, seq=512 (Qwen3-30B-A3B dims):
    L1: lean, Python expert loop
    L2: lean, fused Triton kernel (Ex09)
    D1: dispatch, Python expert loop
    D2: dispatch, fused Triton kernel (Ex09)

Wei's hypothesis: lean's advantage over dispatch widens under the fused
kernel because faster local compute makes the collective-count difference
proportionally larger (Amdahl in the direction lean cares about).
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import torch.distributed as dist


HIDDEN = 2048
INTERMEDIATE = 768
NUM_EXPERTS = 128
TOP_K = 8
NORM_TOPK_PROB = True

BATCH = int(os.environ.get("BATCH", 16))
SEQ = int(os.environ.get("SEQ", 512))

WARMUP = 5
TRIALS = 20
DTYPE = torch.bfloat16


def _time_forward(build: callable, x: torch.Tensor) -> list[float]:
    layer = build()
    for _ in range(WARMUP):
        _ = layer(x)
    torch.cuda.synchronize()
    dist.barrier()
    times_ms: list[float] = []
    for _ in range(TRIALS):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = layer(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    del layer
    torch.cuda.empty_cache()
    return times_ms


def _reduce_max(times_ms: list[float], local_rank: int) -> list[float]:
    t = torch.tensor(times_ms, dtype=torch.float64, device=f"cuda:{local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world_size in (8, 16), f"this bench is TP=EP=world, DP=1 — expected 8 or 16, got {world_size}"

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group("nccl", device_id=torch.device(device))

    from bootcamp.ref.moe import RefSparseMoE
    from bootcamp.ex06_ep.reference import EPSparseMoE as DispatchMoE
    from bootcamp.ex06_ep.reference_lean import EPSparseMoE as LeanMoE
    from bootcamp.ex06_ep.reference_fused import EPSparseMoEDispatchFused as DispatchMoEFused
    from bootcamp.ex06_ep.reference_lean_fused import EPSparseMoELeanFused as LeanMoEFused

    ep_group = dist.group.WORLD  # world = EP group

    # Shared reference — same seed on every rank; used only as weight source.
    torch.manual_seed(42)
    ref = RefSparseMoE(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_experts=NUM_EXPERTS, top_k=TOP_K, norm_topk_prob=NORM_TOPK_PROB,
    ).to(device=device, dtype=DTYPE)

    expert_gate = [ref.experts[e].gate_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_up   = [ref.experts[e].up_proj.weight.data   for e in range(NUM_EXPERTS)]
    expert_down = [ref.experts[e].down_proj.weight.data for e in range(NUM_EXPERTS)]

    def _build(BlockCls):
        def _f():
            layer = BlockCls(
                hidden=HIDDEN, intermediate=INTERMEDIATE,
                num_experts=NUM_EXPERTS, top_k=TOP_K,
                ep_size=world_size, ep_rank=rank, group=ep_group,
                norm_topk_prob=NORM_TOPK_PROB,
            ).to(device=device, dtype=DTYPE)
            layer.weight_loader(
                gate_weight=ref.gate.weight.data,
                expert_gate_weights=expert_gate,
                expert_up_weights=expert_up,
                expert_down_weights=expert_down,
            )
            return layer
        return _f

    # Input: same [B, T, H] on every rank (DP=1 semantics).
    torch.manual_seed(1337)
    x = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=DTYPE)

    if rank == 0:
        print(f"\n===================================================================", flush=True)
        print(f"  Lean vs Dispatch × Python vs Fused   world={world_size}  TP=EP=8 DP=1", flush=True)
        print(f"  H={HIDDEN} I={INTERMEDIATE} E={NUM_EXPERTS} top_k={TOP_K}  dtype={DTYPE}", flush=True)
        print(f"  batch={BATCH} seq={SEQ}  N={BATCH*SEQ}", flush=True)
        print(f"  warmup={WARMUP} trials={TRIALS}", flush=True)
        print(f"===================================================================", flush=True)

    runs = [
        ("L1: lean       + python", LeanMoE),
        ("L2: lean       + fused ", LeanMoEFused),
        ("D1: dispatch   + python", DispatchMoE),
        ("D2: dispatch   + fused ", DispatchMoEFused),
    ]
    medians: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    for label, cls in runs:
        times = _time_forward(_build(cls), x)
        wall = _reduce_max(times, local_rank)
        if rank == 0:
            med = statistics.median(wall)
            sd = statistics.stdev(wall) if len(wall) > 1 else 0.0
            medians[label] = med
            stdevs[label] = sd
            print(f"  {label}:  {med:7.3f} ± {sd:5.3f} ms   (n={len(wall)})", flush=True)

    if rank == 0:
        print(f"\n  === speedups (lean / dispatch, at same compute) ===", flush=True)
        sp_py    = medians["D1: dispatch   + python"] / medians["L1: lean       + python"]
        sp_fused = medians["D2: dispatch   + fused "] / medians["L2: lean       + fused "]
        print(f"  python   :  dispatch / lean = {sp_py:5.3f}x", flush=True)
        print(f"  fused    :  dispatch / lean = {sp_fused:5.3f}x", flush=True)
        print(f"  amplification (fused / python speedup ratio): {sp_fused / sp_py:5.3f}x", flush=True)
        print(f"\n  === speedups (fused / python, at same schedule) ===", flush=True)
        sp_lean = medians["L1: lean       + python"] / medians["L2: lean       + fused "]
        sp_disp = medians["D1: dispatch   + python"] / medians["D2: dispatch   + fused "]
        print(f"  lean     :  python / fused = {sp_lean:5.3f}x", flush=True)
        print(f"  dispatch :  python / fused = {sp_disp:5.3f}x", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
