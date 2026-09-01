#!/usr/bin/env bash
# One question, no clever benchmark: what bandwidth does a plain 16-way
# NCCL allreduce actually deliver across 2 nodes with all defaults on?
# This is the collective Ex07 / lean-schedule / ep=16 actually issues.
#
# Defaults ON:
#   NCCL_P2P (NVLink intra-node) — default enabled
#   NCCL_SHM                     — default enabled
#   NCCL_IB  (RoCE cross-node)   — default enabled
# We do NOT set NCCL_IB_HCA — NCCL picks per-GPU based on NUMA affinity.
set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:-10.134.44.100}"
MASTER_PORT="${MASTER_PORT:-29524}"
WORKSPACE="${WORKSPACE:-/gpfs/users/weiz/workspace/personal/qwen-moe}"
NPROC_LIST="${NPROC_LIST:-2 4 8}"

launch_on() {
    local host="$1" rank="$2" nproc="$3"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "
        cd '$WORKSPACE' &&
        NCCL_DEBUG=WARN NET_MODE=default \
        CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$(($nproc - 1))) \
        uv run python -m torch.distributed.run \
            --nnodes=2 --node_rank=$rank \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            --nproc_per_node=$nproc \
            bootcamp/scripts/nccl_net_bench.py
    "
}

for n in $NPROC_LIST; do
    total=$((2 * n))
    echo "===================================================================="
    echo "  plain allreduce across 2 nodes   nproc/node=$n  world=$total"
    echo "===================================================================="
    launch_on lsf01 1 "$n" > "/tmp/nccl_ex07_rank1_n${n}.log" 2>&1 &
    R1=$!
    sleep 4
    launch_on lsf00 0 "$n" || {
        echo "rank 0 failed. rank 1 tail:" >&2
        tail -30 "/tmp/nccl_ex07_rank1_n${n}.log" >&2
        kill "$R1" 2>/dev/null || true
        continue
    }
    wait "$R1" || echo "(rank 1 exited non-zero; log at /tmp/nccl_ex07_rank1_n${n}.log)"
done
