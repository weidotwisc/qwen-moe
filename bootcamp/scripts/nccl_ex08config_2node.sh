#!/usr/bin/env bash
# Reproduce the exact NCCL config Ex08's launch_multinode.sh uses:
#   TCP transport over net1-0 (100 Gbps netdev, backed by mlx5_0)
#   IB/RDMA disabled → sidesteps the cross-subnet HCA-pairing bug
# and measure world=4/8/16 plain allreduce bandwidth. This is the config
# that actually runs Ex08's full-mesh benchmark.
set -eo pipefail

MASTER_ADDR="${MASTER_ADDR:-10.34.32.48}"   # lsf00's net1-0
MASTER_PORT="${MASTER_PORT:-29525}"
WORKSPACE="${WORKSPACE:-/gpfs/users/weiz/workspace/personal/qwen-moe}"
NPROC_LIST="${NPROC_LIST:-2 4 8}"

launch_on() {
    local host="$1" rank="$2" nproc="$3"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "
        cd '$WORKSPACE' &&
        NET_MODE=ex08-tcp-net1 NCCL_DEBUG=WARN \
        NCCL_SOCKET_IFNAME=net1-0 \
        NCCL_IB_DISABLE=1 \
        CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$(($nproc - 1))) \
        uv run python -m torch.distributed.run \
            --nnodes=2 --node_rank=$rank \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            --nproc_per_node=$nproc \
            bootcamp/scripts/nccl_net_bench.py
    "
}

for n in $NPROC_LIST; do
    echo "===================================================================="
    echo "  world=$((2*n))   nproc/node=$n   NCCL_IB_DISABLE=1  NCCL_SOCKET_IFNAME=net1-0"
    echo "===================================================================="
    launch_on lsf01 1 "$n" > "/tmp/nccl_ex08config_rank1_n${n}.log" 2>&1 &
    R1=$!
    sleep 4
    launch_on lsf00 0 "$n" || {
        echo "rank 0 failed. rank 1 log tail:" >&2
        tail -30 "/tmp/nccl_ex08config_rank1_n${n}.log" >&2
        kill "$R1" 2>/dev/null || true
        continue
    }
    wait "$R1" || echo "(rank 1 exited non-zero)"
done
