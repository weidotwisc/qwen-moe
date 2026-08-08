"""Tests for ex03: QKVParallelLinear + TPMHA."""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.mha import RefMHA
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex03_mha_tp.reference import TPMHA
else:
    from bootcamp.ex03_mha_tp.solution import TPMHA

# n_heads divisible by all of {1, 2, 4, 8}.
HIDDEN, N_HEADS, HEAD_DIM, SEQ, BATCH = 128, 8, 32, 16, 2


def _mha_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    torch.manual_seed(0)
    ref = RefMHA(HIDDEN, N_HEADS, HEAD_DIM).to(device=device, dtype=dtype)
    x = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)

    layer = TPMHA(HIDDEN, N_HEADS, HEAD_DIM, world_size, rank).to(device=device, dtype=dtype)
    layer.qkv_proj.weight_loader(ref.q_proj.weight.data, "q")
    layer.qkv_proj.weight_loader(ref.k_proj.weight.data, "k")
    layer.qkv_proj.weight_loader(ref.v_proj.weight.data, "v")
    layer.o_proj.weight_loader(ref.o_proj.weight.data)

    y = layer(x)
    assert y.shape == (BATCH, SEQ, HIDDEN), f"rank {rank}: got shape {tuple(y.shape)}"

    y_ref = ref(x)
    # Attention accumulates through 2 GEMMs + softmax + o_proj reduction;
    # bf16 numerics drift a bit more than a plain Linear. Bump tolerance.
    scale = 4.0 if dtype == torch.bfloat16 else 1.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_tp_mha(tp_size: int, dtype_str: str) -> None:
    require_gpus(tp_size)
    run_on_ranks(tp_size, _mha_worker, dtype_str)
