"""Correctness tests for Ex09 — Fused MoE Triton kernel (standalone / Scope S).

Single-GPU tests. Compares the Triton kernel's output against
`fused_moe_reference` (Ex05b's per-expert PyTorch compute).

Two axes:
- Config: small (H=128, E=8) + Qwen3-scale (H=2048, E=128).
- Routing: uniform + skewed (adversarial — half the experts get zero records).

Skewed routing verifies the kernel's empty-expert code path
(`offsets[e] == offsets[e+1]`).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bootcamp.dist_utils import require_gpus
from bootcamp.ex09_fused_moe.reference import (
    fused_moe_reference,
    pack_expert_weights,
    prepare_sorted_input,
)
from bootcamp.ex09_fused_moe.solution import fused_moe_forward
from bootcamp.ref.mlp import RefSwiGLU_MLP
from bootcamp.tests.conftest import DTYPES, tol


SMALL_CONFIG = dict(H=128, I=64, E=8, top_k=2, N=64)
QWEN3_CONFIG = dict(H=2048, I=768, E=128, top_k=8, N=1024)


def _run(
    H: int,
    I: int,
    E: int,
    top_k: int,
    N: int,
    dtype: torch.dtype,
    skew: bool,
    tol_scale: float = 8.0,
) -> None:
    require_gpus(1)
    device = "cuda:0"
    torch.manual_seed(0)

    # Build reference model
    experts = nn.ModuleList([RefSwiGLU_MLP(H, I) for _ in range(E)]).to(
        device=device, dtype=dtype
    )
    router_gate = nn.Linear(H, E, bias=False).to(device=device, dtype=dtype)

    # Random input
    torch.manual_seed(1337)
    x = torch.randn(N, H, device=device, dtype=dtype)

    # Route + sort into per-expert contiguous blocks
    prepared = prepare_sorted_input(
        x, top_k=top_k, num_experts=E, router_gate=router_gate, skew=skew
    )
    sorted_x = prepared["sorted_x"]
    offsets = prepared["offsets"]

    # Pack per-expert weights
    packed = pack_expert_weights(experts)
    W_gate, W_up, W_down = packed["W_gate"], packed["W_up"], packed["W_down"]

    # Reference (Python)
    ref_out = fused_moe_reference(sorted_x, offsets, W_gate, W_up, W_down)

    # Triton kernel
    kernel_out = fused_moe_forward(sorted_x, offsets, W_gate, W_up, W_down)

    # Shape sanity
    assert kernel_out.shape == ref_out.shape

    # Correctness — both fp32 and bf16 tolerances key off `tol_scale`
    # (Qwen3-scale accumulates more, so scale up modestly).
    torch.testing.assert_close(kernel_out, ref_out, **tol(dtype, base=tol_scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_small(dtype_str: str, skew: bool) -> None:
    """Small config (H=128, I=64, E=8, top_k=2, N=64) — fast smoke test."""
    _run(**SMALL_CONFIG, dtype=DTYPES[dtype_str], skew=skew, tol_scale=4.0)


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_qwen3_scale(dtype_str: str, skew: bool) -> None:
    """Qwen3-30B-A3B config (H=2048, I=768, E=128, top_k=8, N=1024).

    Larger accumulation → scale tolerance up. Skewed variant activates
    the empty-expert code path since only half the experts receive records.
    """
    _run(**QWEN3_CONFIG, dtype=DTYPES[dtype_str], skew=skew, tol_scale=16.0)
