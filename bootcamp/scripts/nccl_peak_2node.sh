#!/usr/bin/env bash
# One question: can PyTorch NCCL push the RoCE fabric near line rate?
# 2 nodes, 1 GPU each, cross-node allreduce, all NCCL performance knobs on.
set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:-10.134.44.100}"
MASTER_PORT="${MASTER_PORT:-29521}"
WORKSPACE="${WORKSPACE:-/gpfs/users/weiz/workspace/personal/qwen-moe}"

# Every knob that pushes NCCL harder onto the RoCE fabric:
TUNE_ENV="
    NCCL_IB_HCA=mlx5_0,mlx5_3
    NCCL_IB_QPS_PER_CONNECTION=8
    NCCL_IB_SPLIT_DATA_ON_QPS=1
    NCCL_MIN_NCHANNELS=16
    NCCL_MAX_NCHANNELS=32
    NCCL_NTHREADS=512
    NCCL_BUFFSIZE=8388608
    NCCL_NET_GDR_LEVEL=SYS
"

launch_on() {
    local host="$1" rank="$2"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "
        cd '$WORKSPACE' &&
        NET_MODE=roce-tuned NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0 $TUNE_ENV \
        uv run python -m torch.distributed.run \
            --nnodes=2 --node_rank=$rank \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            --nproc_per_node=1 \
            bootcamp/scripts/nccl_net_bench.py
    "
}

launch_on lsf01 1 > /tmp/nccl_peak_rank1.log 2>&1 &
R1=$!
sleep 4
launch_on lsf00 0
wait "$R1"
