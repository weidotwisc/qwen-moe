"""Single-GPU probe for [C9] fused-MoE kernel faults at large row counts.

Context: `MOE_EP_MODE=hybrid MOE_KERNEL=fused` at `TP=1 DP=8` full-scale GSM8K crashes
(CUDA illegal memory access), while `MOE_KERNEL=loop` completes (strict 0.8908) -- so
the fault is isolated to `nanovllm/layers/fused_moe.py` (C9).

IMPORTANT: this script reproduces a LATENT int32 base-pointer overflow -- it faults at
M >~ 1.57M rows (`e_start * stride_xm` exceeds 2**31 with stride == H == 2048). That
would be fixed by widening the base multiply to int64, but that fix is NOT applied
(the kernel is kept verbatim to verified ex09 -- fix + re-verify in ex09, then re-sync).
This is also NOT the same as the actual `TP=1` crash: at TP=1 a rank receives at most
the total dispatched (~1.05M rows), which is BELOW this threshold, and the uniform
M=1.05M case here does NOT crash. The real tp=1 fault is a different, lower-M,
data-pattern-specific bug -- STILL OPEN. To chase it, dump the exact M / offsets /
per-expert counts before the failing `fused_moe_forward` in the real run, reconstruct
that here, and run compute-sanitizer.

Still useful: single-GPU, no mesh/distributed, fast, `compute-sanitizer`-able, and it
checks fused == loop-reference (a silent OOB could corrupt results without crashing):

    CUDA_VISIBLE_DEVICES=0 <venv>/python repro_fused_moe_bug.py
    CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck <venv>/python repro_fused_moe_bug.py

Env knobs to bisect the trigger: E (local experts, def 16), ROWS_PER_E (def 8192),
H (2048), I (768). Set SWEEP=1 to scan ROWS_PER_E and find where it first faults.
"""
import os
import torch
import torch.nn.functional as F

from nanovllm.layers.fused_moe import fused_moe_forward


def loop_reference(x, offsets, Wg, Wu, Wd):
    """C5-b per-expert reference (the known-correct path)."""
    out = torch.empty_like(x)
    for e in range(offsets.numel() - 1):
        s, t = int(offsets[e]), int(offsets[e + 1])
        if s == t:
            continue
        xe = x[s:t]
        out[s:t] = (F.silu(xe @ Wg[e].t()) * (xe @ Wu[e].t())) @ Wd[e].t()
    return out


def run_once(E: int, rows_per_e: int, H: int, I: int, check: bool = True) -> None:
    dev, dt = "cuda", torch.bfloat16
    counts = torch.full((E,), rows_per_e, dtype=torch.long)
    counts[0] += rows_per_e // 2                 # a little imbalance, like real routing
    counts[-1] = max(1, counts[-1] - rows_per_e // 2)
    M = int(counts.sum())
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).to(dev)
    print(f"[repro] E={E} rows/e~{rows_per_e} M={M} H={H} I={I} dtype={dt}", flush=True)

    x = (torch.randn(M, H, device=dev, dtype=dt) * 0.1)
    Wg = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
    Wu = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
    Wd = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02

    y = fused_moe_forward(x, offsets, Wg, Wu, Wd)
    torch.cuda.synchronize()                     # force any async CUDA fault to surface here
    print(f"[repro]   fused OK: y{tuple(y.shape)} finite={bool(torch.isfinite(y).all())}", flush=True)

    if check:
        ref = loop_reference(x, offsets, Wg, Wu, Wd)
        torch.cuda.synchronize()
        d = (y.float() - ref.float()).abs().max().item()
        print(f"[repro]   fused vs loop  max|Δ|={d:.4g}  ({'MATCH' if d < 1e-1 else 'MISMATCH'})", flush=True)


def main():
    torch.manual_seed(0)
    H = int(os.environ.get("H", 2048))
    I = int(os.environ.get("I", 768))
    if os.environ.get("SWEEP", "0") == "1":
        for rpe in (512, 1024, 2048, 4096, 8192, 12288, 16384):
            run_once(int(os.environ.get("E", 16)), rpe, H, I, check=False)
    else:
        run_once(int(os.environ.get("E", 16)), int(os.environ.get("ROWS_PER_E", 8192)), H, I)


if __name__ == "__main__":
    main()
