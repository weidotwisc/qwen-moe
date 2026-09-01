#!/usr/bin/env bash
# Measure aggregate cross-node PyTorch NCCL bandwidth via N concurrent 2-rank
# subgroups. Sweeps NPROC_PER_NODE ∈ {1, 2, 4} unless overridden. Each config:
#   1: 1 concurrent pair    (single GPU pair, single HCA)
#   2: 2 concurrent pairs   (should light up 2 HCAs if NUMA affinities differ)
#   4: 4 concurrent pairs   (2 pairs per HCA)
# Keeps NCCL_P2P / NCCL_SHM DEFAULTS ON — this is the realistic case where
# only cross-node traffic uses RoCE. Pin to mlx5_3 to avoid the cross-subnet
# HCA-pairing bug at higher ranks (see nccl_net_bench_2node.sh).
set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:-10.134.44.100}"
MASTER_PORT="${MASTER_PORT:-29523}"
WORKSPACE="${WORKSPACE:-/gpfs/users/weiz/workspace/personal/qwen-moe}"
# Each entry: "N_pairs:CUDA_VISIBLE_DEVICES". Default picks NUMA-spanning
# GPUs so different pairs get different HCA affinity:
#   NUMA 0 (nvidia-smi topo: GPU 0-3 PIX with mlx5_2/3/4)
#   NUMA 1 (GPU 4-7 PIX with mlx5_0/1)
CONFIGS="${CONFIGS:-1:0  2:0,4  4:0,2,4,6  8:0,1,2,3,4,5,6,7}"

launch_on() {
    local host="$1" rank="$2" nproc="$3" cvd="$4"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "
        cd '$WORKSPACE' &&
        NCCL_DEBUG=WARN \
        NCCL_IB_QPS_PER_CONNECTION=8 \
        NCCL_IB_SPLIT_DATA_ON_QPS=1 \
        NCCL_MIN_NCHANNELS=8 \
        CUDA_VISIBLE_DEVICES=$cvd \
        uv run python -m torch.distributed.run \
            --nnodes=2 --node_rank=$rank \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            --nproc_per_node=$nproc \
            bootcamp/scripts/nccl_aggregate_bench.py
    "
}

for spec in $CONFIGS; do
    n="${spec%%:*}"
    cvd="${spec##*:}"
    echo "================================================================"
    echo "  aggregate bench   N=$n concurrent pairs   world=$((2*n))   GPUs=$cvd"
    echo "================================================================"
    launch_on lsf01 1 "$n" "$cvd" > "/tmp/nccl_agg_rank1_n${n}.log" 2>&1 &
    R1=$!
    sleep 4
    launch_on lsf00 0 "$n" "$cvd" || {
        echo "rank 0 failed. rank 1 log:" >&2
        cat "/tmp/nccl_agg_rank1_n${n}.log" >&2
        kill "$R1" 2>/dev/null || true
        continue
    }
    wait "$R1" || true
done
