"""Cross-node microbenchmark: Ex07 vs Ex08 unfused vs Ex08 fused at TP-4 × DP-4 × EP-16.

Topology (16 ranks across 2 nodes):
    lsf00 (ranks 0-7):
      tp_group_A = {0, 1, 2, 3}   ← intra-node NVLink
      tp_group_B = {4, 5, 6, 7}   ← intra-node NVLink
    lsf01 (ranks 8-15):
      tp_group_C = {8, 9, 10, 11}
      tp_group_D = {12, 13, 14, 15}

    ep_group = {0..15}   ← spans both nodes over RoCE 25GbE

Attention TP-4 AR stays intra-node. MoE dispatch/combine crosses nodes
for inter-TP-group records.

Launched via torchrun:
    # On lsf00:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
        --master_addr=lsf-sshd-main-login-pod --master_port=29900 \\
        bootcamp/ex08_tp_ep_weiz/microbenchmark_multinode.py
    # On lsf01:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \\
        --master_addr=lsf-sshd-main-login-pod --master_port=29900 \\
        bootcamp/ex08_tp_ep_weiz/microbenchmark_multinode.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

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

TP_SIZE = int(os.environ.get("TP_SIZE", 4))
DP_SIZE = int(os.environ.get("DP_SIZE", 4))
EP_SIZE = int(os.environ.get("EP_SIZE", 16))
assert TP_SIZE * DP_SIZE == EP_SIZE, f"TP({TP_SIZE}) × DP({DP_SIZE}) must == EP({EP_SIZE})"

# CONFIGS can be overridden via env, format "b1,s1;b2,s2;..."
_configs_env = os.environ.get("CONFIGS", "")
if _configs_env:
    CONFIGS = [tuple(int(x) for x in c.split(",")) for c in _configs_env.split(";") if c]
else:
    CONFIGS = [
        # Default: B must be divisible by DP_SIZE
        (4, 512), (4, 1024), (4, 2048), (4, 4096), (4, 8192),
        (8, 512), (8, 1024), (8, 2048),
        (16, 512), (16, 1024),
        (32, 512), (64, 512),
    ]

WARMUP = int(os.environ.get("WARMUP", 3))
TRIALS = int(os.environ.get("TRIALS", 15))
DTYPE = torch.bfloat16


def _time_block(
    BlockCls, ref, rank, tp_rank, tp_group, ep_rank, ep_group,
    batch, seq, dp_start, dp_end, device,
) -> list[float]:
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


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    assert world_size == 16, f"Cross-node benchmark requires 16 ranks, got {world_size}"

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{local_rank}"

    # Build DP_SIZE TP groups of TP_SIZE ranks each. Contiguous rank ordering:
    # (TP=4, DP=4): 4 groups {0-3},{4-7},{8-11},{12-15} — each intra-node.
    # (TP=8, DP=2): 2 groups {0-7},{8-15} — each intra-node (one per node).
    tp_groups = [
        dist.new_group(ranks=list(range(g * TP_SIZE, (g + 1) * TP_SIZE)))
        for g in range(DP_SIZE)
    ]
    my_tp_idx = rank // TP_SIZE
    tp_group = tp_groups[my_tp_idx]
    tp_rank = rank % TP_SIZE
    dp_rank = my_tp_idx
    ep_group = None  # world
    ep_rank = rank

    from bootcamp.ref.block import RefBlock
    from bootcamp.ex07_tp_ep_hybrid.reference import HybridBlock as Ex07Block
    from bootcamp.ex08_tp_ep_weiz.reference import HybridScheduleBlock as Ex08Block
    from bootcamp.ex08_tp_ep_weiz.reference_fused import HybridScheduleBlockFused as Ex08FusedBlock

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
            continue

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
        print("\n" + "=" * 118, flush=True)
        print(f"Qwen3-30B-A3B block @ TP-{TP_SIZE} × DP-{DP_SIZE} × EP-{EP_SIZE} (multi-node), dtype={DTYPE}", flush=True)
        print(f"  H={HIDDEN} I={INTERMEDIATE} E={NUM_EXPERTS} top_k={TOP_K}", flush=True)
        print(f"  warmup={WARMUP} trials={TRIALS}", flush=True)
        print("=" * 118, flush=True)
        print(
            f"{'batch':>6}{'seq':>6}{'N_tp':>7}"
            f"{'ex07 (ms)':>16}{'ex08 unfused':>19}{'ex08 fused':>19}"
            f"{'08/07':>8}{'08f/07':>9}",
            flush=True,
        )
        print("-" * 118, flush=True)
        for r in results:
            print(
                f"{r['batch']:>6}{r['seq']:>6}{r['N_tp']:>7}"
                f"{r['ex07_ms']:>11.2f}±{r['ex07_std']:>3.1f}"
                f"{r['ex08_ms']:>13.2f}±{r['ex08_std']:>3.1f}"
                f"{r['ex08f_ms']:>13.2f}±{r['ex08f_std']:>3.1f}"
                f"{r['speedup_08']:>7.2f}x{r['speedup_08f']:>8.2f}x",
                flush=True,
            )
        print("=" * 118, flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
