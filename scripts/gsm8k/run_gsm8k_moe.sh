#!/usr/bin/env bash
# GSM8K accuracy A/B for the Qwen3-MoE integration: loop (C5) vs fused (C9).
# Single GPU; runs on any node sharing the GPFS (e.g. lsf01). Uses GPU 0.
#
#   scripts/gsm8k/run_gsm8k_moe.sh              # full 1319-question test set
#   LIMIT=100 scripts/gsm8k/run_gsm8k_moe.sh    # quick 100-question first pass
#
# GSM8K is loaded directly by gsm8k_moe.py (the repo .venv has `datasets`).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REPO_VENV="${REPO_VENV:-$REPO/.venv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOGS="$REPO/scripts/gsm8k/logs"; mkdir -p "$LOGS"
TAG="${LIMIT:-full}"   # log suffix: the #questions (e.g. 100) or "full" -> no clobber across sizes

cd "$REPO/nanovllm-weiz"
for k in loop fused; do
  echo "=========================================================="
  echo "  GSM8K  MOE_KERNEL=$k   (GPU $CUDA_VISIBLE_DEVICES, LIMIT=${LIMIT:-all})"
  echo "=========================================================="
  MOE_KERNEL="$k" LIMIT="${LIMIT:-}" \
    "$REPO_VENV/bin/python" gsm8k_moe.py 2>&1 | tee "$LOGS/gsm8k_${k}_${TAG}.log"
done

echo; echo "===== summary (oracle: vLLM strict ~0.8923) ====="
grep -h "^\[gsm8k\] kernel=" "$LOGS/gsm8k_loop_${TAG}.log" "$LOGS/gsm8k_fused_${TAG}.log"
