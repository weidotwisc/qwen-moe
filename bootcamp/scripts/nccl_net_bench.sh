#!/usr/bin/env bash
# Single-node NCCL bandwidth bench across three transports on the same host.
# Uses torchrun --standalone so its launch shape matches nccl_net_bench_2node.sh.
#
# NCCL picks a transport per (src, dst) pair based on availability, in priority:
#   1. NVLink / PCIe P2P            (governed by NCCL_P2P_DISABLE)
#   2. Host shared memory           (governed by NCCL_SHM_DISABLE, same-host only)
#   3. RDMA verbs / IB / RoCE       (governed by NCCL_IB_DISABLE + NCCL_IB_HCA)
#   4. TCP over sockets             (governed by NCCL_SOCKET_IFNAME)
#
# Turning off higher-priority transports forces NCCL down the ladder.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NPROC="${NPROC:-2}"

run_mode() {
    local mode="$1"
    export NET_MODE="$mode"
    unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE
    unset NCCL_SOCKET_IFNAME NCCL_IB_HCA

    case "$mode" in
        nvlink) : ;;
        roce)
            export NCCL_P2P_DISABLE=1
            export NCCL_SHM_DISABLE=1
            export NCCL_IB_HCA="mlx5_0,mlx5_3"
            ;;
        tcp)
            export NCCL_P2P_DISABLE=1
            export NCCL_SHM_DISABLE=1
            export NCCL_IB_DISABLE=1
            export NCCL_SOCKET_IFNAME=eth0
            ;;
        *) echo "unknown mode: $mode" >&2; exit 1 ;;
    esac

    uv run python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node="$NPROC" \
        bootcamp/scripts/nccl_net_bench.py
}

for mode in nvlink roce tcp; do
    run_mode "$mode"
done
