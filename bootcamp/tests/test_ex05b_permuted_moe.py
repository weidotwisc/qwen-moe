"""Tests for ex05b — permuted MoE with grouped compute.

Compares against `RefSparseMoE` (bootcamp/ref/moe.py). The permuted
implementation should produce the same output up to fp reduction-order
noise — the accumulation order differs between the two implementations,
so tolerance is slightly looser than ex05a's.

Single-GPU tests — no mp.spawn / NCCL setup.
"""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.ref.moe import RefSparseMoE
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex05_moe_baseline.reference_b import PermutedSparseMoE
else:
    from bootcamp.ex05_moe_baseline.solution_b import PermutedSparseMoE


HIDDEN = 128
INTERMEDIATE = 64
NUM_EXPERTS = 8
TOP_K = 2
N_TOKENS = 32
BATCH = 2
SEQ = N_TOKENS // BATCH


def _copy_weights(dst: PermutedSparseMoE, src: RefSparseMoE) -> None:
    """Copy weights from a RefSparseMoE into a PermutedSparseMoE instance."""
    dst.gate.weight.data.copy_(src.gate.weight.data)
    for e in range(NUM_EXPERTS):
        dst.experts[e].gate_proj.weight.data.copy_(src.experts[e].gate_proj.weight.data)
        dst.experts[e].up_proj.weight.data.copy_(src.experts[e].up_proj.weight.data)
        dst.experts[e].down_proj.weight.data.copy_(src.experts[e].down_proj.weight.data)


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
def test_permuted_sparse_moe(dtype_str: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    dtype = DTYPES[dtype_str]
    device = "cuda:0"

    torch.manual_seed(0)
    ref = RefSparseMoE(HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, norm_topk_prob=True).to(
        device=device, dtype=dtype
    )

    layer = PermutedSparseMoE(HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, norm_topk_prob=True).to(
        device=device, dtype=dtype
    )
    _copy_weights(layer, ref)

    x = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)

    y = layer(x)
    y_ref = ref(x)

    # Permuted vs naive accumulate in different orders → slightly looser than 05a.
    scale = 5.0 if dtype == torch.bfloat16 else 3.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))
