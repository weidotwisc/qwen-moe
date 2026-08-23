"""Tests for Ex07 — TP-4 × DP-2 × EP-8 hybrid block.

Topology:
- 8 GPUs total.
- Two TP groups: {0,1,2,3} and {4,5,6,7}.
- One EP group over all 8 ranks.
- Each TP group processes a distinct half of the batch (DP-2 outer).

Every rank generates the SAME global x[N, H] via seeded torch.manual_seed().
Each TP group takes its half of the batch (replicated across the group's 4 ranks).
Rank r's output is compared to the corresponding half of RefBlock(x)[N, H].
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.block import RefBlock
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex07_tp_ep_hybrid.reference import HybridBlock
else:
    from bootcamp.ex07_tp_ep_hybrid.solution import HybridBlock


# Test-scale config (matches Qwen3-30B-A3B structure at smaller dims).
HIDDEN = 128
INTERMEDIATE = 64
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 32
NUM_EXPERTS = 8
TOP_K = 2
BATCH = 2
SEQ = 16
N_TOKENS = BATCH * SEQ  # 32 tokens globally

TP_SIZE = 4  # 2 TP groups on 8 GPUs
DP_SIZE = 2  # 2 DP replicas (== number of TP groups)
EP_SIZE = 8  # EP over all 8


def _hybrid_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    assert world_size == 8, "Ex07 requires exactly 8 GPUs for TP-4 × DP-2 × EP-8"

    # Construct process groups: two TP groups + one EP group (= world).
    tp_group_a = dist.new_group(ranks=[0, 1, 2, 3])
    tp_group_b = dist.new_group(ranks=[4, 5, 6, 7])
    tp_group = tp_group_a if rank < 4 else tp_group_b
    tp_rank = rank % TP_SIZE
    dp_rank = rank // TP_SIZE  # which DP replica (0 = group A, 1 = group B)
    ep_group = None  # world group — every rank participates
    ep_rank = rank

    # Every rank builds an identical RefBlock (via seed) to serve as the oracle.
    torch.manual_seed(0)
    ref = RefBlock(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K, norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

    # Build the HybridBlock on each rank.
    layer = HybridBlock(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K,
        tp_size=TP_SIZE, tp_rank=tp_rank, tp_group=tp_group,
        ep_size=EP_SIZE, ep_rank=ep_rank, ep_group=ep_group,
        norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

    # Load weights from ref → layer.
    expert_gate_weights = [ref.moe.experts[e].gate_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_up_weights = [ref.moe.experts[e].up_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_down_weights = [ref.moe.experts[e].down_proj.weight.data for e in range(NUM_EXPERTS)]
    layer.weight_loader(
        attn_norm_weight=ref.attn_norm.weight.data,
        q_weight=ref.attn.q_proj.weight.data,
        k_weight=ref.attn.k_proj.weight.data,
        v_weight=ref.attn.v_proj.weight.data,
        o_weight=ref.attn.o_proj.weight.data,
        moe_norm_weight=ref.moe_norm.weight.data,
        gate_weight=ref.moe.gate.weight.data,
        expert_gate_weights=expert_gate_weights,
        expert_up_weights=expert_up_weights,
        expert_down_weights=expert_down_weights,
    )

    # Every rank generates the SAME global x, then slices its DP replica's half.
    # DP replica A (ranks 0-3) sees the first N/2 tokens; replica B (4-7) the second.
    assert BATCH % DP_SIZE == 0, "Batch must be divisible by DP_SIZE"
    local_B = BATCH // DP_SIZE
    torch.manual_seed(1337)
    x_full = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)
    dp_start = dp_rank * local_B
    dp_end = dp_start + local_B
    local_x = x_full[dp_start:dp_end]  # [local_B, SEQ, HIDDEN] — same for all ranks in this TP group

    # Forward on the hybrid block.
    y = layer(local_x)

    # Oracle: run RefBlock on the full batch, slice out our DP replica's half.
    y_ref_full = ref(x_full)
    y_ref = y_ref_full[dp_start:dp_end]

    assert y.shape == (local_B, SEQ, HIDDEN), f"rank {rank}: got shape {tuple(y.shape)}"

    # Tolerance: block accumulates through GQA (2 matmul + softmax + o_proj AR) +
    # MoE (router + expert compute + weighted sum + all_reduce/gather). Bump scale.
    scale = 8.0 if dtype == torch.bfloat16 else 5.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
def test_hybrid_block(dtype_str: str) -> None:
    require_gpus(8)
    run_on_ranks(8, _hybrid_worker, dtype_str)
