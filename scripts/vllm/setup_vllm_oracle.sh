#!/usr/bin/env bash
# One-time setup: an ISOLATED venv for the vLLM GSM8K accuracy oracle.
#
# Why a separate venv: the project .venv is deliberately pinned to
# torch 2.11+cu126 for the nano-vLLM work. vLLM pins its own torch, so mixing
# them would break one or the other. This box's driver is r580 / CUDA 13.0, so
# whatever CUDA wheel vLLM pulls (cu12x/cu13x) will run.
set -euo pipefail
cd "$(dirname "$0")"   # scripts/vllm/ — keep the vLLM env self-contained here, away from the repo-root uv project
VENV="${VENV:-.venv-vllm}"

uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" "lm-eval[vllm]"

echo "== installed =="
"$VENV/bin/python" - <<'PY'
import vllm, lm_eval
print("vllm    ", vllm.__version__)
print("lm_eval ", lm_eval.__version__)
PY
echo "Done. Run the oracle with: scripts/vllm/run_gsm8k_vllm.sh"
