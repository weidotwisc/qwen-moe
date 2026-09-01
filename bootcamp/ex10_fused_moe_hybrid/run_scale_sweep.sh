#!/usr/bin/env bash
# Ex07 vs Ex10 microbenchmark across three scales:
#   world=4   TP=2 DP=2 EP=4   single-node
#   world=8   TP=4 DP=2 EP=8   single-node
#   world=16  TP=8 DP=2 EP=16  two-node (TCP over net1-0, per Ex08 config)
#
# Same Qwen3-30B-A3B block dims across all three, so numbers are
# apples-to-apples across scales.
set -eo pipefail

WORKSPACE="${WORKSPACE:-/gpfs/users/weiz/workspace/personal/qwen-moe}"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-$WORKSPACE/bootcamp/ex10_fused_moe_hybrid/results}"
mkdir -p "$BENCH_OUT_DIR"

SCRIPT="$WORKSPACE/bootcamp/ex10_fused_moe_hybrid/microbenchmark.py"

single_node_run() {
    local nproc="$1" tp="$2" dp="$3" ep="$4"
    echo "======================================================"
    echo "  single-node: world=$nproc  TP=$tp DP=$dp EP=$ep"
    echo "======================================================"
    cd "$WORKSPACE"
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc - 1))) \
    NCCL_DEBUG=WARN \
    TP_SIZE=$tp DP_SIZE=$dp EP_SIZE=$ep \
    BENCH_OUT_DIR="$BENCH_OUT_DIR" \
    uv run python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=$nproc \
        "$SCRIPT"
}

two_node_run() {
    local nproc_per=8 tp=8 dp=2 ep=16
    local master_addr="10.134.44.100"
    local master_port=29601
    echo "======================================================"
    echo "  two-node: world=$((2*nproc_per))  TP=$tp DP=$dp EP=$ep"
    echo "  master=$master_addr:$master_port  transport=TCP/net1-0"
    echo "======================================================"

    launch() {
        local host="$1" node_rank="$2"
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "
            cd '$WORKSPACE' &&
            NCCL_DEBUG=WARN \
            NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=net1-0 \
            TP_SIZE=$tp DP_SIZE=$dp EP_SIZE=$ep \
            BENCH_OUT_DIR='$BENCH_OUT_DIR' \
            CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc_per - 1))) \
            uv run python -m torch.distributed.run \
                --nnodes=2 --node_rank=$node_rank \
                --master_addr=$master_addr --master_port=$master_port \
                --nproc_per_node=$nproc_per \
                '$SCRIPT'
        "
    }

    launch lsf01 1 > "$BENCH_OUT_DIR/rank1_w16.log" 2>&1 &
    R1=$!
    sleep 4
    launch lsf00 0 || {
        echo "rank0 failed; rank1 log:" >&2
        tail -40 "$BENCH_OUT_DIR/rank1_w16.log" >&2
        kill "$R1" 2>/dev/null || true
        return 1
    }
    wait "$R1" || echo "(rank1 exited non-zero; see $BENCH_OUT_DIR/rank1_w16.log)"
}

# Scope selector — pass "single" / "two" / "all" (default all)
SCOPE="${1:-all}"
case "$SCOPE" in
    single|all)
        single_node_run 4 2 2 4
        single_node_run 8 4 2 8
        ;;
esac
case "$SCOPE" in
    two|all)
        two_node_run
        ;;
esac

echo
echo "Results written to $BENCH_OUT_DIR/*.json"
