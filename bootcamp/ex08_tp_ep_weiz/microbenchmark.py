"""Microbenchmark: Ex07 canonical vs Ex08 hybrid schedule at TP-4 × DP-2 × EP-8.

Measures per-block wall-clock time for both:
- `HybridBlock` (Ex07): canonical dispatch + within-TP all_gather.
- `HybridScheduleBlock` (Ex08): intra-lean + inter-dispatch + within-TP all_reduce.

Both blocks include attention TP-4 + MoE + RMSNorm + residuals, using
Qwen3-30B-A3B dims. Same weights loaded into both. Same input.

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python bootcamp/ex08_tp_ep_weiz/microbenchmark.py

Output: table of per-config median times for both blocks, plus the empirical
speedup Ex07 → Ex08.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
import os

import torch
import torch.distributed as dist

from bootcamp.dist_utils import run_on_ranks


# ============================== Qwen3-30B-A3B config ==============================
HIDDEN = 2048
INTERMEDIATE = 768
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = 128
NUM_EXPERTS = 128
TOP_K = 8
NORM_TOPK_PROB = True

# TP-4 × DP-2 × EP-8 topology
TP_SIZE = 4
DP_SIZE = 2
EP_SIZE = 8

# Configs to sweep — (batch, seq_len). batch must be divisible by DP_SIZE.
CONFIGS: list[tuple[int, int]] = [
    # Long-context sweep (B=2, growing SEQ)
    (2, 512),
    (2, 1024),
    (2, 2048),
    (2, 4096),
    (2, 8192),
    (2, 16384),
    # Large-batch sweep (constant SEQ=512, growing B)
    (8, 512),
    (16, 512),
    (32, 512),
    (64, 512),
    # Mixed
    (8, 1024),
    (8, 2048),
    (8, 4096),
    (16, 1024),
    (16, 2048),
    (32, 1024),
]

WARMUP = 3
TRIALS = 20
DTYPE = torch.bfloat16


def _time_block(
    BlockCls, ref, rank, tp_rank, tp_group, ep_rank, ep_group,
    batch, seq, dp_start, dp_end, device,
) -> list[float]:
    """Build a block, load weights, warmup, time TRIALS forwards. Returns ms."""
    layer = BlockCls(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K,
        tp_size=TP_SIZE, tp_rank=tp_rank, tp_group=tp_group,
        ep_size=EP_SIZE, ep_rank=ep_rank, ep_group=ep_group,
        norm_topk_prob=NORM_TOPK_PROB,
    ).to(device=device, dtype=DTYPE)

    expert_gate_weights = [ref.moe.experts[e].gate_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_up_weights = [ref.moe.experts[e].up_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_down_weights = [ref.moe.experts[e].down_proj.weight.data for e in range(NUM_EXPERTS)]
    layer.weight_loader(
        attn_norm_weight=ref.attn_norm.weight.data,
        q_weight=ref.attn.q_proj.weight.data,
        k_weight=ref.attn.k_proj.weight.data,
        v_weight=ref.attn.v_proj.weight.data,
        o_weight=ref.attn.o_proj.weight.data,
        moe_norm_weight=ref.moe_norm.weight.data,
        gate_weight=ref.moe.gate.weight.data,
        expert_gate_weights=expert_gate_weights,
        expert_up_weights=expert_up_weights,
        expert_down_weights=expert_down_weights,
    )

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
    device = f"cuda:{torch.cuda.current_device()}"
    t = torch.tensor(times_ms, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def _bench_worker(rank: int, world_size: int, results_file: str) -> None:
    assert world_size == 8, "Ex08 benchmark requires 8 GPUs (TP-4 × DP-2 × EP-8)"
    device = f"cuda:{torch.cuda.current_device()}"

    tp_group_a = dist.new_group(ranks=[0, 1, 2, 3])
    tp_group_b = dist.new_group(ranks=[4, 5, 6, 7])
    tp_group = tp_group_a if rank < 4 else tp_group_b
    tp_rank = rank % TP_SIZE
    dp_rank = rank // TP_SIZE
    ep_group = None
    ep_rank = rank

    from bootcamp.ref.block import RefBlock
    from bootcamp.ex07_tp_ep_hybrid.reference import HybridBlock as Ex07Block
    from bootcamp.ex08_tp_ep_weiz.reference import HybridScheduleBlock as Ex08Block
    from bootcamp.ex08_tp_ep_weiz.reference_fused import HybridScheduleBlockFused as Ex08FusedBlock

    # Build one RefBlock per rank as the weight source (identical on every rank via seed).
    torch.manual_seed(42)
    ref = RefBlock(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K, norm_topk_prob=NORM_TOPK_PROB,
    ).to(device=device, dtype=DTYPE)

    results: list[dict] = []

    for batch, seq in CONFIGS:
        if batch < DP_SIZE:
            continue
        assert batch % DP_SIZE == 0
        local_B = batch // DP_SIZE
        dp_start = dp_rank * local_B
        dp_end = dp_start + local_B

        N_tp = local_B * seq
        if N_tp % TP_SIZE != 0:
            continue  # skip if within-TP striping wouldn't be clean

        if rank == 0:
            print(f"\n--- batch={batch} seq={seq} (local_B={local_B}, N_tp={N_tp}) ---",
                  flush=True)

        ex07_times = _time_block(
            Ex07Block, ref, rank, tp_rank, tp_group, ep_rank, ep_group,
            batch, seq, dp_start, dp_end, device,
        )
        ex08_times = _time_block(
            Ex08Block, ref, rank, tp_rank, tp_group, ep_rank, ep_group,
            batch, seq, dp_start, dp_end, device,
        )
        ex08f_times = _time_block(
            Ex08FusedBlock, ref, rank, tp_rank, tp_group, ep_rank, ep_group,
            batch, seq, dp_start, dp_end, device,
        )

        ex07_wall = _reduce_max(ex07_times)
        ex08_wall = _reduce_max(ex08_times)
        ex08f_wall = _reduce_max(ex08f_times)

        if rank == 0:
            e07_med = statistics.median(ex07_wall)
            e08_med = statistics.median(ex08_wall)
            e08f_med = statistics.median(ex08f_wall)
            e07_std = statistics.stdev(ex07_wall) if len(ex07_wall) > 1 else 0.0
            e08_std = statistics.stdev(ex08_wall) if len(ex08_wall) > 1 else 0.0
            e08f_std = statistics.stdev(ex08f_wall) if len(ex08f_wall) > 1 else 0.0
            speedup_08 = e07_med / e08_med
            speedup_08f = e07_med / e08f_med
            results.append({
                "batch": batch, "seq": seq, "local_B": local_B, "N_tp": N_tp,
                "ex07_ms": e07_med, "ex07_std": e07_std,
                "ex08_ms": e08_med, "ex08_std": e08_std,
                "ex08f_ms": e08f_med, "ex08f_std": e08f_std,
                "speedup_08": speedup_08,
                "speedup_08f": speedup_08f,
            })
            print(
                f"  ex07: {e07_med:6.2f}±{e07_std:.1f}ms | "
                f"ex08: {e08_med:6.2f}±{e08_std:.1f}ms ({speedup_08:5.2f}x) | "
                f"ex08_fused: {e08f_med:6.2f}±{e08f_std:.1f}ms ({speedup_08f:5.2f}x)",
                flush=True,
            )

    if rank == 0:
        with open(results_file, "w") as f:
            json.dump(results, f)


def _print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 118, flush=True)
    print(f"Qwen3-30B-A3B block @ TP-4 × DP-2 × EP-8, dtype={DTYPE}", flush=True)
    print(f"  H={HIDDEN} I={INTERMEDIATE} E={NUM_EXPERTS} top_k={TOP_K}", flush=True)
    print(f"  n_heads={N_HEADS} n_kv_heads={N_KV_HEADS} head_dim={HEAD_DIM}", flush=True)
    print(f"  warmup={WARMUP} trials={TRIALS} (max-reduced across ranks)", flush=True)
    print("=" * 118, flush=True)
    print(
        f"{'batch':>6}{'seq':>6}{'N_tp':>7}"
        f"{'ex07 (ms)':>16}{'ex08 unfused':>19}{'ex08 fused':>19}"
        f"{'08/07':>8}{'08f/07':>9}",
        flush=True,
    )
    print("-" * 118, flush=True)
    for r in all_results:
        print(
            f"{r['batch']:>6}{r['seq']:>6}{r['N_tp']:>7}"
            f"{r['ex07_ms']:>11.2f}±{r['ex07_std']:>3.1f}"
            f"{r['ex08_ms']:>13.2f}±{r['ex08_std']:>3.1f}"
            f"{r['ex08f_ms']:>13.2f}±{r['ex08f_std']:>3.1f}"
            f"{r['speedup_08']:>7.2f}x{r['speedup_08f']:>8.2f}x",
            flush=True,
        )
    print("=" * 118, flush=True)


def main() -> int:
    max_gpus = torch.cuda.device_count()
    if max_gpus < 8:
        print(f"Need at least 8 GPUs for TP-4 × DP-2 × EP-8, have {max_gpus}.",
              file=sys.stderr)
        return 1

    all_results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        results_file = os.path.join(tmp, "bench_ex07_vs_ex08.json")
        run_on_ranks(8, _bench_worker, results_file)
        with open(results_file) as f:
            all_results.extend(json.load(f))

    _print_summary(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
