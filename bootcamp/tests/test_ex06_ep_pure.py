"""Tests for Ex06 pure EP — DP-partitioned inputs, dispatch load-bearing.

Each rank owns a distinct 1/ep_size slice of a global batch. The layer
processes only its own slice; dispatch moves tokens to expert-owning
ranks; combine returns results. Each rank's output is verified against
the corresponding slice of a single-GPU RefSparseMoE reference run.

Parametrized over ep_size ∈ {1, 2, 4, 8} × dtype ∈ {fp32, bf16}.
"""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.moe import RefSparseMoE
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex06_ep_pure.reference import EPSparseMoE
else:
    from bootcamp.ex06_ep_pure.solution import EPSparseMoE


HIDDEN = 128
INTERMEDIATE = 64
NUM_EXPERTS = 8
TOP_K = 2
BATCH = 2
SEQ = 16
N_TOKENS = BATCH * SEQ    # 32 tokens globally


def _ep_pure_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    # Same seed on every rank → identical reference module + identical global x.
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

    # Every rank generates the SAME global x (via seed), then slices out its
    # own share. This gives us a global oracle for the test: rank r's local_y
    # should equal ref(x_full)[r's slice].
    assert N_TOKENS % world_size == 0
    local_N = N_TOKENS // world_size
    x_full = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)
    x_full_flat = x_full.reshape(N_TOKENS, HIDDEN)

    # Rank r's slice.
    local_x_flat = x_full_flat[rank * local_N : (rank + 1) * local_N]  # [local_N, H]
    local_x = local_x_flat.reshape(BATCH // world_size if BATCH >= world_size else 1,
                                   local_N // max(1, BATCH // world_size),
                                   HIDDEN)

    local_y = layer(local_x)  # rank r's output for its own tokens
    local_y_flat = local_y.reshape(-1, HIDDEN)

    # Compare rank r's output against ref(x_full)[r's slice].
    y_ref_flat = ref(x_full).reshape(N_TOKENS, HIDDEN)
    y_ref_local = y_ref_flat[rank * local_N : (rank + 1) * local_N]

    scale = 5.0 if dtype == torch.bfloat16 else 3.0
    torch.testing.assert_close(local_y_flat, y_ref_local, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
def test_ep_sparse_moe_pure(ep_size: int, dtype_str: str) -> None:
    require_gpus(ep_size)
    run_on_ranks(ep_size, _ep_pure_worker, dtype_str)
