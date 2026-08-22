"""Tests for Ex06 lean variant — one all_reduce per forward.

Same fixture as test_ex06_ep.py, different import target. Both variants
should produce the same output vs `RefSparseMoE` up to fp reduction-order
noise.

Parametrized over `ep_size ∈ {1, 2, 4, 8}` and `dtype ∈ {fp32, bf16}`.
"""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.moe import RefSparseMoE
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex06_ep.reference_lean import EPSparseMoE
else:
    from bootcamp.ex06_ep.solution_lean import EPSparseMoE


# Test config — matches test_ex06_ep.py for direct comparison.
HIDDEN = 128
INTERMEDIATE = 64
NUM_EXPERTS = 8
TOP_K = 2
BATCH = 2
SEQ = 16
N_TOKENS = BATCH * SEQ


def _ep_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    torch.manual_seed(0)

    ref = RefSparseMoE(HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, norm_topk_prob=True).to(
        device=device, dtype=dtype
    )

    layer = EPSparseMoE(
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        ep_size=world_size,
        ep_rank=rank,
        group=None,
        norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

    expert_gate_weights = [ref.experts[e].gate_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_up_weights = [ref.experts[e].up_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_down_weights = [ref.experts[e].down_proj.weight.data for e in range(NUM_EXPERTS)]
    layer.weight_loader(
        gate_weight=ref.gate.weight.data,
        expert_gate_weights=expert_gate_weights,
        expert_up_weights=expert_up_weights,
        expert_down_weights=expert_down_weights,
    )

    x = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)

    y = layer(x)
    y_ref = ref(x)

    assert y.shape == (BATCH, SEQ, HIDDEN), f"rank {rank}: got shape {tuple(y.shape)}"

    # Lean variant's all_reduce sum is a slightly different reduction order
    # than the reference's per-expert loop, but well within fp tolerance.
    scale = 5.0 if dtype == torch.bfloat16 else 3.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
def test_ep_sparse_moe_lean(ep_size: int, dtype_str: str) -> None:
    require_gpus(ep_size)
    run_on_ranks(ep_size, _ep_worker, dtype_str)
