"""Tests for Ex10 — Fused hybrid block (Ex09 kernel inside Ex07 HybridBlock).

Reuses Ex07's test harness. The composition claim is:

    FusedHybridBlock.forward(x) == HybridBlock.forward(x)

up to declared floating-point tolerance, on every rank. Both blocks are
compared against the same single-GPU RefBlock oracle, so passing this
test also validates the composition against the reference.
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
    from bootcamp.ex10_fused_moe_hybrid.reference import FusedHybridBlock
else:
    from bootcamp.ex10_fused_moe_hybrid.solution import FusedHybridBlock

HIDDEN = 128
INTERMEDIATE = 64
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 32
NUM_EXPERTS = 8
TOP_K = 2
BATCH = 2
SEQ = 16

TP_SIZE = 4
DP_SIZE = 2
EP_SIZE = 8


def _fused_hybrid_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"
    assert world_size == 8

    tp_group_a = dist.new_group(ranks=[0, 1, 2, 3])
    tp_group_b = dist.new_group(ranks=[4, 5, 6, 7])
    tp_group = tp_group_a if rank < 4 else tp_group_b
    tp_rank = rank % TP_SIZE
    dp_rank = rank // TP_SIZE
    ep_group = None
    ep_rank = rank

    torch.manual_seed(0)
    ref = RefBlock(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K, norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

    layer = FusedHybridBlock(
        hidden=HIDDEN, intermediate=INTERMEDIATE,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K,
        tp_size=TP_SIZE, tp_rank=tp_rank, tp_group=tp_group,
        ep_size=EP_SIZE, ep_rank=ep_rank, ep_group=ep_group,
        norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

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

    local_B = BATCH // DP_SIZE
    torch.manual_seed(1337)
    x_full = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)
    dp_start = dp_rank * local_B
    dp_end = dp_start + local_B
    local_x = x_full[dp_start:dp_end]

    y = layer(local_x)
    y_ref = ref(x_full)[dp_start:dp_end]

    assert y.shape == (local_B, SEQ, HIDDEN)

    scale = 8.0 if dtype == torch.bfloat16 else 5.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
def test_fused_hybrid_block(dtype_str: str) -> None:
    require_gpus(8)
    run_on_ranks(8, _fused_hybrid_worker, dtype_str)
