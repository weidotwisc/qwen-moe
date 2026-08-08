"""Tests for ex02: MergedColumnParallelLinear + TPSwiGLUMLP."""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.mlp import RefSwiGLU_MLP
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex02_mlp_tp.reference import TPSwiGLUMLP
else:
    from bootcamp.ex02_mlp_tp.solution import TPSwiGLUMLP

HIDDEN, INTERMEDIATE, SEQ = 128, 64, 8


def _mlp_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    torch.manual_seed(0)
    ref = RefSwiGLU_MLP(HIDDEN, INTERMEDIATE).to(device=device, dtype=dtype)
    x = torch.randn(SEQ, HIDDEN, device=device, dtype=dtype)

    layer = TPSwiGLUMLP(HIDDEN, INTERMEDIATE, world_size, rank).to(device=device, dtype=dtype)
    # Load gate, up, down from the reference.
    layer.gate_up_proj.weight_loader(ref.gate_proj.weight.data, 0)
    layer.gate_up_proj.weight_loader(ref.up_proj.weight.data, 1)
    layer.down_proj.weight_loader(ref.down_proj.weight.data)

    y = layer(x)
    assert y.shape == (SEQ, HIDDEN), f"rank {rank}: got shape {tuple(y.shape)}"

    y_ref = ref(x)
    # SwiGLU MLP accumulates through 2 matmuls + reduce → a bit more numerical
    # drift than a single Linear; bump the bf16 tolerance slightly.
    scale = 2.0 if dtype == torch.bfloat16 else 1.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_tp_swiglu_mlp(tp_size: int, dtype_str: str) -> None:
    require_gpus(tp_size)
    run_on_ranks(tp_size, _mlp_worker, dtype_str)
