#!/usr/bin/env python
"""Ex11 — the DP lockstep invariant + the "lockstep tax" (microbenchmark).

torchrun micro-benchmark, no nano-vLLM import. Shares the toy machinery with the
exercise: infra in `common.py`, the two cores in `reference.py`. Distils the one
transferable idea from nano-vLLM's central-orchestration data parallelism:

    ep_group == world  =>  a cross-replica all-to-all sits in EVERY forward, so
    all DP replicas must co-arrive at every layer. A replica cannot skip a step
    (e.g. when its scheduler is momentarily empty) or the world-wide collective
    blocks forever. The fix is the *dummy forward*: an idle replica runs a
    token-less forward purely to post its share of the collective schedule.

Two things this measures that we could not otherwise guess:

  --mode invariant : correctness. With the dummy forward the run completes; with
                     --no-dummy an idle replica skips the collective and the group
                     DEADLOCKS (a wall-clock watchdog self-reports + hard-exits the
                     wedged process, since a CUDA-stream hang can't be try/excepted).

  --mode tax       : the "lockstep tax". Because every replica co-arrives per step,
                     the SLOWEST replica gates the step. Ideal (independent) DP wall
                     = max_r Sum_s c[s,r]; lockstep wall = Sum_s max_r c[s,r].
                     tax = lockstep / ideal >= 1, driven by per-step load imbalance
                     and the idle->dummy fraction. Reported vs a sweep of both.

Real reference: nano-vLLM's MoE block posts 3 all-to-alls per layer over
ep_group=world (split-negotiate + dispatch + combine), see
bootcamp/ex06_ep_pure/reference.py:117-153. This toy keeps the load-independent
split-negotiation all_to_all (a tiny [world] exchange) as the co-arrival barrier,
and folds the load-dependent work into an O(n) MLP -- so the tax lands on compute.

Progress + human tables -> stderr; one JSONL row -> stdout (so `> file` is clean).

Launch (8-GPU single node, TP=1 DP=8):
    TP_SIZE=1 torchrun --nproc_per_node=8 \\
        bootcamp/ex11_dp_lockstep/dp_lockstep_bench.py --mode tax
Or use bootcamp/ex11_dp_lockstep/run.sh for the standard demos + sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from bootcamp.ex11_dp_lockstep.common import (
    ToyLayer, build_mesh, install_watchdog, make_schedule, reduce_max_scalar, run_layers,
)
from bootcamp.ex11_dp_lockstep.reference import lockstep_tax


def log(*a, **k):
    """Human-readable progress -> stderr, rank-0 only."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, file=sys.stderr, flush=True, **k)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["invariant", "tax"], default="tax")
    p.add_argument("--no-dummy", action="store_true",
                   help="[invariant] idle replica SKIPS the collective -> deadlock")
    p.add_argument("--hidden", type=int, default=2048)      # Qwen3-30B-A3B H
    p.add_argument("--ffn-mult", type=int, default=2,       # expert MLP width = mult*H
                   help="per-layer compute is an O(n) MLP H->mult*H->H")
    p.add_argument("--layers", type=int, default=16)        # toy depth (real: 48)
    p.add_argument("--steps", type=int, default=64)         # [tax] global steps
    p.add_argument("--base-tokens", type=int, default=2048) # [tax] mean batch/replica
    p.add_argument("--imbalance", type=float, default=0.5,  # [tax] CV of per-step batch
                   help="per-step batch ~ base*(1 + imbalance*N(0,1)), clamped >=1")
    p.add_argument("--idle-frac", type=float, default=0.0,  # [tax] P(replica idle in a step)
                   help="fraction of (step,replica) cells that are idle -> dummy forward")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout", type=float, default=None,
                   help="process-group timeout (s); default 20 invariant / 1800 tax")
    return p.parse_args()


# ----------------------------------------------------------------- invariant
def run_invariant(args, mesh) -> int:
    device = f"cuda:{mesh.local_rank}"
    layers = [ToyLayer(args.hidden, args.ffn_mult, mesh.world, device) for _ in range(args.layers)]

    idle_replica = mesh.dp_size - 1                 # deterministic: last replica idle
    is_idle = (mesh.replica == idle_replica)
    dummy = not args.no_dummy
    rank = mesh.rank

    log(f"[invariant] world={mesh.world} TP={mesh.tp_size} DP={mesh.dp_size} "
        f"layers={args.layers} dummy={dummy}  (idle replica = r{idle_replica})")

    def _deadlock_verdict():
        log(f"  RESULT: DEADLOCK CONFIRMED — no progress for {args.timeout:.0f}s.\n"
            f"           idle replica r{idle_replica} never posted the all_to_all; "
            f"busy ranks blocked on it.")
        if rank == 0:
            print(json.dumps({
                "mode": "invariant", "world": mesh.world, "tp": mesh.tp_size,
                "dp": mesh.dp_size, "layers": args.layers, "dummy": dummy,
                "outcome": "deadlock",
            }), flush=True)
        os._exit(0)   # abandon the wedged NCCL comm; OS reclaims the GPUs

    watchdog = install_watchdog(args.timeout, _deadlock_verdict)

    if is_idle and not dummy:
        # THE BUG: an empty scheduler skips its forward -> never posts the world
        # all_to_all. The busy replicas block on it forever (watchdog fires).
        log(f"  rank{rank} (idle replica) SKIPPING collective schedule (--no-dummy)")
    else:
        n = 0 if is_idle else args.base_tokens      # idle -> dummy (1 row)
        run_layers(layers, n, collective=True)
    dist.barrier()                                  # final co-arrival
    torch.cuda.synchronize()

    watchdog.cancel()                               # completed in time
    log("  RESULT: COMPLETED — every replica posted the collective schedule.")
    if rank == 0:
        print(json.dumps({
            "mode": "invariant", "world": mesh.world, "tp": mesh.tp_size,
            "dp": mesh.dp_size, "layers": args.layers, "dummy": dummy,
            "outcome": "completed",
        }), flush=True)
    return 0


# ----------------------------------------------------------------------- tax
def run_tax(args, mesh) -> int:
    device = f"cuda:{mesh.local_rank}"
    layers = [ToyLayer(args.hidden, args.ffn_mult, mesh.world, device) for _ in range(args.layers)]
    sched = make_schedule(args.steps, mesh.dp_size, args.base_tokens,
                          args.imbalance, args.idle_frac, args.seed)
    my_batches = [sched[s][mesh.replica] for s in range(args.steps)]   # this replica's stream

    def run_pass(collective: bool) -> tuple[float, list[float]]:
        """Return (total wall on this rank, per-step compute time list)."""
        per_step = []
        if collective:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for n in my_batches:
            ts0 = time.perf_counter()
            run_layers(layers, n, collective=collective)
            if not collective:                    # per-step compute only (no sync)
                torch.cuda.synchronize()
            per_step.append(time.perf_counter() - ts0)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, per_step

    for _ in range(args.warmup):
        run_pass(collective=True)

    lockstep_walls, ideal_walls, last_csteps = [], [], None
    for _ in range(args.trials):
        w_lock, _ = run_pass(collective=True)                       # real lockstep
        w_ideal, csteps = run_pass(collective=False)                # independent DP
        lockstep_walls.append(reduce_max_scalar(w_lock, device))    # slowest rank gates
        ideal_walls.append(reduce_max_scalar(w_ideal, device))
        last_csteps = csteps

    lockstep_wall = statistics.median(lockstep_walls)
    ideal_wall = statistics.median(ideal_walls)

    # gather per-step compute times: [world, steps] -> per replica (max over TP) -> [dp, steps]
    c_local = torch.tensor(last_csteps, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(c_local) for _ in range(mesh.world)]
    dist.all_gather(gathered, c_local)

    tokens = sum(sum(row) for row in sched)         # total token-forwards (all replicas)
    if mesh.rank == 0:
        c = torch.stack(gathered).reshape(mesh.dp_size, mesh.tp_size, args.steps).amax(dim=1)
        tax = lockstep_tax(c)                        # shared with the exercise (reference.py)
        tax_structural = tax["tax"]
        tax_wall = lockstep_wall / ideal_wall if ideal_wall else float("nan")
        overhead_frac = max(0.0, (lockstep_wall - tax["sum_max"]) / lockstep_wall)
        row = {
            "mode": "tax", "world": mesh.world, "tp": mesh.tp_size, "dp": mesh.dp_size,
            "layers": args.layers, "hidden": args.hidden, "steps": args.steps,
            "base_tokens": args.base_tokens, "imbalance": args.imbalance,
            "idle_frac": args.idle_frac, "trials": args.trials,
            "tokens": tokens,
            "lockstep_wall_s": round(lockstep_wall, 4),
            "ideal_wall_s": round(ideal_wall, 4),
            "tax_wall": round(tax_wall, 4),
            "tax_structural": round(tax_structural, 4),
            "collective_overhead_frac": round(overhead_frac, 4),
            "lockstep_tok_s": round(tokens / lockstep_wall) if lockstep_wall else None,
            "ideal_tok_s": round(tokens / ideal_wall) if ideal_wall else None,
        }
        log(f"[tax] imbalance={args.imbalance:<4} idle_frac={args.idle_frac:<4} "
            f"| tax(wall)={tax_wall:5.2f}x  tax(structural)={tax_structural:5.2f}x  "
            f"a2a_overhead={overhead_frac*100:4.1f}%  "
            f"| lockstep {tokens/lockstep_wall:8.0f} tok/s  ideal {tokens/ideal_wall:8.0f} tok/s")
        print(json.dumps(row), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.timeout is None:
        args.timeout = 20.0 if args.mode == "invariant" else 1800.0
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    dist.init_process_group(
        "nccl",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        timeout=timedelta(seconds=args.timeout),
    )
    try:
        mesh = build_mesh()
        rc = run_invariant(args, mesh) if args.mode == "invariant" else run_tax(args, mesh)
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
