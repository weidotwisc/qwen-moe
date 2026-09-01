#!/usr/bin/env bash
# Cross-node NCCL bandwidth bench: launch torchrun on lsf00 (rank 0) and
# lsf01 (rank 1) via SSH, no LSF scheduler needed.
#
# GPFS workspace is visible on both hosts, so no code sync is needed.
# eth0 is used only for TCP rendezvous — the NCCL data plane picks its own
# transport based on the same env-var knobs as the single-node bench:
#   roce → NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1 + NCCL_IB_HCA=mlx5_0,mlx5_3
#   tcp  → also NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME=eth0
# ("nvlink" mode is meaningless cross-host — no NVLink between hosts.)
#
# Set NPROC_PER_NODE=8 to run a full 16-GPU cross-node allreduce (both hosts
# have 8 A100s). Default is 1 GPU per node for a clean single-QP-per-HCA
# comparison against ib_write_bw numbers.

set -euo pipefail

HOST0="${HOST0:-lsf00}"
HOST1="${HOST1:-lsf01}"
MASTER_ADDR="${MASTER_ADDR:-10.134.44.100}"  # lsf00's eth0
MASTER_PORT="${MASTER_PORT:-29520}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
WORKSPACE="${WORKSPACE:-/gpfs/users/weiz/workspace/personal/qwen-moe}"

env_for_mode() {
    # The fabric has TWO independent L2 subnets, one per HCA:
    #   mlx5_0 / net1-0 → 10.34.32.0/23
    #   mlx5_3 / net1-1 → 10.34.34.0/23
    # RoCEv2 is not routable across them. Letting NCCL use BOTH HCAs
    # (NCCL_IB_HCA=mlx5_0,mlx5_3) makes it produce cross-subnet QP pairs
    # at world_size≥16 that time out in ibv_modify_qp. Pin to one HCA
    # so every pair stays intra-subnet. Halves theoretical aggregate
    # bandwidth to ~12 GB/s but is stable.
    case "$1" in
        roce) echo "NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_3}" ;;
        tcp)  echo "NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0" ;;
        *) echo "unknown mode: $1" >&2; exit 1 ;;
    esac
}

launch_on() {
    local host="$1" node_rank="$2" mode="$3"
    local envs
    envs=$(env_for_mode "$mode")
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "
        cd '$WORKSPACE' && \
        NET_MODE=$mode NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1))) \
        $envs \
        uv run python -m torch.distributed.run \
            --nnodes=2 --node_rank=$node_rank \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            --nproc_per_node=$NPROC_PER_NODE \
            bootcamp/scripts/nccl_net_bench.py
    "
}

for mode in roce tcp; do
    echo "===================================================================="
    echo "  2-node mode=$mode   $HOST0 (rank0)  <->  $HOST1 (rank1)   "
    echo "  master=$MASTER_ADDR:$MASTER_PORT   nproc_per_node=$NPROC_PER_NODE"
    echo "===================================================================="

    # rank 1 first, in background (writes to log — rank 0 output is the main channel)
    launch_on "$HOST1" 1 "$mode" > "/tmp/nccl_net_bench_rank1_${mode}.log" 2>&1 &
    R1_PID=$!
    sleep 4  # let rank 1 reach the rendezvous port

    # rank 0 in foreground — its stdout is the main output
    launch_on "$HOST0" 0 "$mode" || {
        echo "rank 0 failed — rank 1 log follows:" >&2
        cat "/tmp/nccl_net_bench_rank1_${mode}.log" >&2
        kill "$R1_PID" 2>/dev/null || true
        exit 1
    }
    wait "$R1_PID" || echo "(rank 1 exited non-zero; see /tmp/nccl_net_bench_rank1_${mode}.log)"
done
