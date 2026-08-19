"""Tests for Ex06 — Expert Parallelism (EP).

Spawns `ep_size` processes; each runs `EPSparseMoE` on the same replicated
input tensor. Compares the (all-gathered) output against single-GPU
`RefSparseMoE`.

Parametrized over `ep_size ∈ {1, 2, 4, 8}` and `dtype ∈ {fp32, bf16}`.

Tolerance is looser than Ex05a because EP's accumulation order differs
from the reference (contributions arrive in different orders after
all_to_all combine).
"""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.moe import RefSparseMoE
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex06_ep.reference import EPSparseMoE
else:
    from bootcamp.ex06_ep.solution import EPSparseMoE


# Test config — small enough to run fast, structured to exercise all cases.
HIDDEN = 128
INTERMEDIATE = 64
NUM_EXPERTS = 8            # divisible by all tested ep_sizes {1, 2, 4, 8}
TOP_K = 2
BATCH = 2
SEQ = 16
N_TOKENS = BATCH * SEQ     # 32, divisible by 8 (max ep_size)


def _ep_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    # Seed identically on every rank → same reference constructed, same random input.
    torch.manual_seed(0)

    # Reference lives on rank r's GPU too so we can compare locally.
    ref = RefSparseMoE(HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, norm_topk_prob=True).to(
        device=device, dtype=dtype
    )

    # Build the EP layer for this rank.
    layer = EPSparseMoE(
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        ep_size=world_size,
        ep_rank=rank,
        group=None,          # default world group; Ex07 will pass a sub-group here
        norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

    # Load weights: full gate + only this rank's experts.
    expert_gate_weights = [ref.experts[e].gate_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_up_weights = [ref.experts[e].up_proj.weight.data for e in range(NUM_EXPERTS)]
    expert_down_weights = [ref.experts[e].down_proj.weight.data for e in range(NUM_EXPERTS)]
    layer.weight_loader(
        gate_weight=ref.gate.weight.data,
        expert_gate_weights=expert_gate_weights,
        expert_up_weights=expert_up_weights,
        expert_down_weights=expert_down_weights,
    )

    # Replicated input: same seed on every rank → identical tensor.
    x = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)

    # Forward on both.
    y = layer(x)
    y_ref = ref(x)

    assert y.shape == (BATCH, SEQ, HIDDEN), f"rank {rank}: got shape {tuple(y.shape)}"

    # EP output must match reference up to fp reduction-order noise.
    # Bump scale for bf16 — combine sums k contributions per token, each of
    # which itself sums through 3 matmuls in the expert MLP.
    scale = 5.0 if dtype == torch.bfloat16 else 3.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
def test_ep_sparse_moe(ep_size: int, dtype_str: str) -> None:
    require_gpus(ep_size)
    run_on_ranks(ep_size, _ep_worker, dtype_str)
