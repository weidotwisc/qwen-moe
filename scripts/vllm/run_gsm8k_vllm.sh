#!/usr/bin/env bash
# GSM8K accuracy ORACLE via lm-eval-harness with the vLLM backend.
# 5-shot, greedy, reports strict-match and flexible-extract exact-match
# (the same protocol the paper uses).
#
# Usage:
#   scripts/run_gsm8k_vllm.sh                                   # full 30B, TP=8
#   MODEL=Qwen/Qwen3-0.6B TP=1 LIMIT=5 scripts/run_gsm8k_vllm.sh  # fast smoke test
#
# Env knobs (all optional):
#   MODEL   HF id or local path         (default Qwen/Qwen3-30B-A3B, already cached)
#   TP      tensor_parallel_size        (default 8)
#   LIMIT   #examples (blank = full 1319; e.g. 5 or 50 for a quick slice)
#   MAXLEN  max_model_len               (default 4096)
#   OUT     output dir                  (default results/gsm8k_<model>_tp<TP>)
set -euo pipefail
cd "$(dirname "$0")"   # scripts/vllm/ — env + results live here, self-contained
VENV="${VENV:-.venv-vllm}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
TP="${TP:-8}"
LIMIT="${LIMIT:-}"
MAXLEN="${MAXLEN:-4096}"
OUT="${OUT:-results/gsm8k_$(echo "$MODEL" | tr '/' '_')_tp${TP}}"

# Cache-first by default, but allow a download if something is missing
# (the smoke-test 0.6B may only be partially cached). Set OFFLINE=1 to forbid network.
[[ "${OFFLINE:-0}" == "1" ]] && export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$OUT"
LIMIT_ARG=(); [[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

set -x
"$VENV/bin/lm_eval" --model vllm \
  --model_args "pretrained=${MODEL},tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=${MAXLEN},enforce_eager=True,safetensors_load_strategy=prefetch" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size auto \
  "${LIMIT_ARG[@]}" \
  --output_path "$OUT" \
  --log_samples
set +x
echo "== results written under $OUT =="
