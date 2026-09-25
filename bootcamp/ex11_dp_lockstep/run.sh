#!/usr/bin/env bash
# Ex11 — DP lockstep invariant + lockstep-tax driver.
#
#   ./run.sh invariant   the correctness demo: dummy forward completes; --no-dummy DEADLOCKS
#   ./run.sh tax         the lockstep-tax sweep (imbalance x idle_frac), JSONL -> results/
#   ./run.sh all         both (default)
#
# Env: NPROC (default 8), TP_SIZE (default 1 => DP=NPROC, the C7 shape).
set -eo pipefail

WORKSPACE="${WORKSPACE:-/gpfs/users/weiz/workspace/personal/qwen-moe}"
SCRIPT="$WORKSPACE/bootcamp/ex11_dp_lockstep/dp_lockstep_bench.py"
OUT_DIR="${OUT_DIR:-$WORKSPACE/bootcamp/ex11_dp_lockstep/results}"
mkdir -p "$OUT_DIR"
cd "$WORKSPACE"

NPROC="${NPROC:-8}"
TP_SIZE="${TP_SIZE:-1}"

# DP=8 spawns ~NPROC procs; cap BLAS threads or torch trips
# "OpenBLAS pthread_create: Resource temporarily unavailable" at init.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"

launch() {  # launch <extra bench args...>
    TP_SIZE="$TP_SIZE" uv run python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node="$NPROC" "$SCRIPT" "$@"
}

demo_invariant() {
    echo "=== [1/2] dummy forward (idle replica still posts the collective) ===" >&2
    launch --mode invariant --layers 8 --timeout 15

    echo >&2
    echo "=== [2/2] --no-dummy (idle replica SKIPS the collective -> expect DEADLOCK) ===" >&2
    echo "    the bench self-reports + hard-exits after a 15s watchdog; OS timeout is backup." >&2
    timeout --signal=KILL 60 bash -c "$(declare -f launch); \
        TP_SIZE='$TP_SIZE' NPROC='$NPROC' SCRIPT='$SCRIPT' \
        launch --mode invariant --layers 8 --no-dummy --timeout 15" \
        || echo "  (torchrun exited non-zero — expected for the deadlock case)" >&2
    pkill -9 -u "$USER" -f dp_lockstep_bench 2>/dev/null || true
    sleep 2
}

sweep_tax() {
    local out="$OUT_DIR/tax_tp${TP_SIZE}_dp$((NPROC / TP_SIZE)).jsonl"
    : > "$out"
    echo "=== lockstep-tax sweep: world=$NPROC TP=$TP_SIZE DP=$((NPROC / TP_SIZE)) ===" >&2
    echo "    JSONL -> $out" >&2
    for imb in 0.0 0.25 0.5 1.0; do
        for idle in 0.0 0.25 0.5; do
            # NCCL prints "NCCL version ..." to stdout; keep only the JSONL row.
            launch --mode tax --imbalance "$imb" --idle-frac "$idle" \
                --steps 64 --layers 16 --base-tokens 2048 --trials 3 \
                | grep -E '^\{' >> "$out"
        done
    done
    echo >&2
    echo "=== summary ($out) ===" >&2
    uv run python - "$out" >&2 <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
print(f"{'imbal':>6}{'idle':>6}{'tax_wall':>10}{'tax_struct':>12}{'a2a_ovh%':>10}"
      f"{'lock tok/s':>12}{'ideal tok/s':>13}")
for r in rows:
    print(f"{r['imbalance']:>6}{r['idle_frac']:>6}{r['tax_wall']:>9.2f}x"
          f"{r['tax_structural']:>11.2f}x{r['collective_overhead_frac']*100:>9.1f}%"
          f"{r['lockstep_tok_s']:>12}{r['ideal_tok_s']:>13}")
PY
}

case "${1:-all}" in
    invariant) demo_invariant ;;
    tax)       sweep_tax ;;
    all)       demo_invariant; sweep_tax ;;
    *) echo "usage: $0 [invariant|tax|all]" >&2; exit 2 ;;
esac
