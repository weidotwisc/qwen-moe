"""Microbenchmark: dispatch vs lean EP at Qwen3-30B-A3B scale.

Measures per-forward wall-clock time for both `reference.py` and
`reference_lean.py` on the actual Qwen3-30B-A3B MoE dimensions
(hidden=2048, intermediate=768, num_experts=128, top_k=8).

Sweeps over N (batch × seq_len) to observe the crossover behavior:
predicted lean advantage at small N (fewer collectives dominate),
narrowing at large N (transfer dominates).

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python bootcamp/ex06_ep/microbenchmark.py

Output: table of median forward times per configuration for each variant,
plus the empirical speedup ratio.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
import os
import sys

import torch
import torch.distributed as dist

from bootcamp.dist_utils import run_on_ranks


# ============================== Qwen3-30B-A3B config ==============================
HIDDEN = 2048
INTERMEDIATE = 768
NUM_EXPERTS = 128
TOP_K = 8
NORM_TOPK_PROB = True

# Configs to sweep — (batch, seq_len). N = batch × seq_len must be divisible by ep_size.
CONFIGS: list[tuple[int, int]] = [
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (1, 8192),
    (8, 1024),      # medium batch
    (32, 512),      # large-batch decode-ish
]

WARMUP = 5
TRIALS = 30
DTYPE = torch.bfloat16


class _HotExpertRouter(torch.nn.Module):
    """A deterministic router that always routes to a fixed set of experts.

    Regardless of input `x`, emits identical logits per token. `hot_experts`
    get a high logit; all others get a low logit. Combined with `topk`, this
    guarantees every token routes to the hot experts.

    Used to stress-test the "popular expert" load-imbalance pathology of
    dispatch-based EP.
    """

    def __init__(self, num_experts: int, hot_experts: list[int]) -> None:
        super().__init__()
        # High logit for hot experts, low for others. Softmax will pick these.
        logits = torch.full((num_experts,), -100.0, dtype=torch.float32)
        for e in hot_experts:
            logits[e] = 100.0
        # Fake a .weight for downstream compatibility (some code inspects it).
        self.register_buffer("_fixed_logits", logits)
        # nn.Linear-compat: expose .weight so weight_loader-like code paths don't break.
        self.weight = torch.nn.Parameter(
            torch.zeros(num_experts, 1), requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]
        # Broadcast fixed logits to [N, num_experts], match dtype.
        return self._fixed_logits.to(x.dtype).unsqueeze(0).expand(N, -1).contiguous()


def _time_variant(
    LayerCls, rank: int, world_size: int, batch: int, seq: int, device: str,
    hot_experts: list[int] | None = None,
) -> list[float]:
    """Instantiate a variant, warm up, time TRIALS forwards. Returns per-trial ms.

    If `hot_experts` is provided, override the router so every token routes to
    exactly those experts (via top-k picking the highest logits). This creates
    maximum load imbalance if `hot_experts` are all on one rank.
    """
    torch.manual_seed(42)  # identical weights across variants
    layer = LayerCls(
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        ep_size=world_size,
        ep_rank=rank,
        group=None,
        norm_topk_prob=NORM_TOPK_PROB,
    ).to(device=device, dtype=DTYPE)

    if hot_experts is not None:
        layer.gate = _HotExpertRouter(NUM_EXPERTS, hot_experts).to(device=device, dtype=DTYPE)

    torch.manual_seed(1337)  # identical input across variants
    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=DTYPE)

    # Warmup
    for _ in range(WARMUP):
        _ = layer(x)
    torch.cuda.synchronize()
    dist.barrier()

    # Time
    times_ms: list[float] = []
    for _ in range(TRIALS):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = layer(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    # Free before next variant runs.
    del layer, x
    torch.cuda.empty_cache()
    return times_ms


def _reduce_max(times_ms: list[float]) -> list[float]:
    """Take per-trial max across ranks (wall-clock is bounded by slowest)."""
    t = torch.tensor(times_ms, dtype=torch.float64, device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def _bench_worker(rank: int, world_size: int, results_file: str, mode: str) -> None:
    """mode: 'uniform' (default routing) or 'skew' (all top_k experts on rank 0)."""
    # run_on_ranks or _torchrun_main already initialized the process group and
    # bound this process to a local CUDA device. Use whichever device is current.
    local_device = torch.cuda.current_device()
    device = f"cuda:{local_device}"

    from bootcamp.ex06_ep.reference import EPSparseMoE as DispatchEP
    from bootcamp.ex06_ep.reference_lean import EPSparseMoE as LeanEP

    # Under 'skew' mode: force every token to route to experts [0, 1, ..., top_k-1].
    # These all live on rank 0 (which owns experts [0, experts_per_rank)).
    hot_experts: list[int] | None = None
    if mode == "skew":
        hot_experts = list(range(TOP_K))

    results: list[dict] = []

    for batch, seq in CONFIGS:
        N = batch * seq
        if N % world_size != 0:
            continue

        if rank == 0:
            print(
                f"\n--- batch={batch} seq={seq} N={N} ep={world_size} mode={mode} ---",
                flush=True,
            )

        dispatch_times = _time_variant(
            DispatchEP, rank, world_size, batch, seq, device, hot_experts=hot_experts
        )
        lean_times = _time_variant(
            LeanEP, rank, world_size, batch, seq, device, hot_experts=hot_experts
        )

        dispatch_wall = _reduce_max(dispatch_times)
        lean_wall = _reduce_max(lean_times)

        if rank == 0:
            d_med = statistics.median(dispatch_wall)
            l_med = statistics.median(lean_wall)
            d_std = statistics.stdev(dispatch_wall) if len(dispatch_wall) > 1 else 0.0
            l_std = statistics.stdev(lean_wall) if len(lean_wall) > 1 else 0.0
            speedup = d_med / l_med
            results.append({
                "ep": world_size, "mode": mode,
                "batch": batch, "seq": seq, "N": N,
                "dispatch_ms": d_med, "dispatch_std": d_std,
                "lean_ms": l_med, "lean_std": l_std,
                "speedup": speedup,
            })
            print(
                f"  dispatch: {d_med:6.2f} ± {d_std:5.2f} ms  |  "
                f"lean: {l_med:6.2f} ± {l_std:5.2f} ms  |  "
                f"lean speedup: {speedup:5.2f}x",
                flush=True,
            )

    if rank == 0:
        with open(results_file, "w") as f:
            json.dump(results, f)


def _print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 100, flush=True)
    print(f"Qwen3-30B-A3B config @ dtype={DTYPE}", flush=True)
    print(f"  H={HIDDEN} I={INTERMEDIATE} E={NUM_EXPERTS} top_k={TOP_K}", flush=True)
    print(f"  warmup={WARMUP} trials={TRIALS} (max-reduced across ranks)", flush=True)
    print("=" * 100, flush=True)
    print(
        f"{'mode':>8}{'ep':>4}{'batch':>6}{'seq':>7}{'N':>9}"
        f"{'dispatch (ms)':>18}{'lean (ms)':>16}{'speedup':>11}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for r in all_results:
        print(
            f"{r['mode']:>8}{r['ep']:>4}{r['batch']:>6}{r['seq']:>7}{r['N']:>9}"
            f"{r['dispatch_ms']:>13.2f} ± {r['dispatch_std']:>3.1f}"
            f"{r['lean_ms']:>11.2f} ± {r['lean_std']:>3.1f}"
            f"{r['speedup']:>9.2f}x",
            flush=True,
        )
    print("=" * 100, flush=True)


def _torchrun_main() -> int:
    """Multi-node entry: launched by `torchrun`, world_size is pre-set."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    modes_env = os.environ.get("MODES", "uniform,skew")
    modes = [m.strip() for m in modes_env.split(",")]

    all_results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for mode in modes:
            results_file = os.path.join(tmp, f"bench_ep{world_size}_{mode}.json")
            if rank == 0:
                print(f"\n{'#' * 60}", flush=True)
                print(f"# ep_size = {world_size}, routing mode = {mode}", flush=True)
                print(f"{'#' * 60}", flush=True)
            _bench_worker(rank, world_size, results_file, mode)
            if rank == 0:
                with open(results_file) as f:
                    all_results.extend(json.load(f))

    if rank == 0:
        _print_summary(all_results)

    dist.destroy_process_group()
    return 0


def main() -> int:
    # Multi-node / torchrun path: RANK is set by the launcher.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return _torchrun_main()

    # Single-node path: mp.spawn ourselves.
    max_gpus = torch.cuda.device_count()
    if max_gpus < 2:
        print(f"Need at least 2 GPUs, have {max_gpus}.", file=sys.stderr)
        return 1

    ep_sizes_env = os.environ.get("EP_SIZES", "8")  # focus on ep=8 for skew study
    ep_sizes = [int(s) for s in ep_sizes_env.split(",") if int(s) <= max_gpus]
    if not ep_sizes:
        print(f"No valid ep_sizes in {ep_sizes_env} for max_gpus={max_gpus}.", file=sys.stderr)
        return 1

    modes_env = os.environ.get("MODES", "uniform,skew")
    modes = [m.strip() for m in modes_env.split(",")]

    all_results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for ep_size in ep_sizes:
            for mode in modes:
                results_file = os.path.join(tmp, f"bench_ep{ep_size}_{mode}.json")
                print(f"\n{'#' * 60}", flush=True)
                print(f"# ep_size = {ep_size}, routing mode = {mode}", flush=True)
                print(f"{'#' * 60}", flush=True)
                run_on_ranks(ep_size, _bench_worker, results_file, mode)
                with open(results_file) as f:
                    all_results.extend(json.load(f))

    _print_summary(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
