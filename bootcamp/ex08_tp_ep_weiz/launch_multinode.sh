#!/bin/bash
# Cross-node benchmark launcher for Ex08 comparison.
# Runs on lsf00 (node_rank=0) — SSHs into lsf01 (node_rank=1) and starts it
# in the background, then runs the local half here in the foreground.
#
# Both nodes share GPFS, so we launch from the repo path and write the
# per-node log to a shared results dir.

set -eo pipefail

# --- config ---
MASTER_ADDR=${MASTER_ADDR:-10.34.32.48}     # lsf00's net1-0 interface
MASTER_PORT=${MASTER_PORT:-29901}
NCCL_IFACE=${NCCL_IFACE:-net1-0}
REPO=/gpfs/users/weiz/workspace/personal/qwen-moe
PY=$REPO/.venv/bin/python
SCRIPT=$REPO/bootcamp/ex08_tp_ep_weiz/microbenchmark_multinode.py
RESULTS=$REPO/bootcamp/ex08_tp_ep_weiz/results_multinode
mkdir -p "$RESULTS"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG0=$RESULTS/rank0_${TIMESTAMP}.log
LOG1=$RESULTS/rank1_${TIMESTAMP}.log

echo "[launcher] master=$MASTER_ADDR:$MASTER_PORT iface=$NCCL_IFACE"
echo "[launcher] log rank0: $LOG0"
echo "[launcher] log rank1: $LOG1"

# Pass-through knobs for the workload (topology + configs + trials).
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-4}
EP_SIZE=${EP_SIZE:-16}
CONFIGS=${CONFIGS:-}
WARMUP=${WARMUP:-3}
TRIALS=${TRIALS:-15}

echo "[launcher] topo TP=$TP_SIZE DP=$DP_SIZE EP=$EP_SIZE trials=$TRIALS"
echo "[launcher] CONFIGS=$CONFIGS"

# --- start lsf01 (node_rank=1) in the background via SSH ---
ssh lsf01 "cd $REPO && \
    PYTHONPATH=$REPO \
    NCCL_SOCKET_IFNAME=$NCCL_IFACE \
    NCCL_DEBUG=WARN \
    NCCL_IB_DISABLE=1 \
    TP_SIZE=$TP_SIZE DP_SIZE=$DP_SIZE EP_SIZE=$EP_SIZE \
    CONFIGS='$CONFIGS' WARMUP=$WARMUP TRIALS=$TRIALS \
    $PY -m torch.distributed.run \
        --nnodes=2 --nproc_per_node=8 --node_rank=1 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        $SCRIPT > $LOG1 2>&1" &
SSH_PID=$!

# Give lsf01's rendezvous a head start.
sleep 3

# --- run lsf00 (node_rank=0) in the foreground here ---
cd $REPO
PYTHONPATH=$REPO \
NCCL_SOCKET_IFNAME=$NCCL_IFACE \
NCCL_DEBUG=WARN \
NCCL_IB_DISABLE=1 \
TP_SIZE=$TP_SIZE DP_SIZE=$DP_SIZE EP_SIZE=$EP_SIZE \
CONFIGS="$CONFIGS" WARMUP=$WARMUP TRIALS=$TRIALS \
$PY -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    $SCRIPT 2>&1 | tee "$LOG0"
RC=${PIPESTATUS[0]}

echo "[launcher] rank0 exit=$RC — waiting for rank1 SSH to complete"
wait $SSH_PID || echo "[launcher] rank1 SSH exit=$?"

echo "[launcher] done"
exit $RC
