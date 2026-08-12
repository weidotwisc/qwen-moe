"""Tests for ex04: QKVParallelLinearGQA + TPGQA.

Includes the KV-replication case: at tp_size=8 with n_kv_heads=4, pairs of
ranks share the same KV head.
"""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ref.gqa import RefGQA
from bootcamp.tests.conftest import DTYPES, tol

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex04_gqa_tp.reference import TPGQA
else:
    from bootcamp.ex04_gqa_tp.solution import TPGQA

# GQA config: 8 Q heads, 4 KV heads (2:1 ratio, mirrors Qwen3's structure at
# smaller scale). At tp_size ∈ {1, 2, 4} the KV heads shard cleanly.
# At tp_size = 8, num_kv_heads=4 < tp_size=8 → KV replication (2 replicas).
HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM, SEQ, BATCH = 128, 8, 4, 32, 16, 2


def _gqa_worker(rank: int, world_size: int, dtype_str: str) -> None:
    dtype = DTYPES[dtype_str]
    device = f"cuda:{rank}"

    torch.manual_seed(0)  # identical ref on every rank
    ref = RefGQA(HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM).to(device=device, dtype=dtype)
    x = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)

    layer = TPGQA(HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM, world_size, rank).to(
        device=device, dtype=dtype
    )
    layer.qkv_proj.weight_loader(ref.q_proj.weight.data, "q")
    layer.qkv_proj.weight_loader(ref.k_proj.weight.data, "k")
    layer.qkv_proj.weight_loader(ref.v_proj.weight.data, "v")
    layer.o_proj.weight_loader(ref.o_proj.weight.data)

    y = layer(x)
    assert y.shape == (BATCH, SEQ, HIDDEN), f"rank {rank}: got shape {tuple(y.shape)}"

    y_ref = ref(x)
    # GQA has the same numerical characteristics as MHA — attention accumulates
    # through 2 GEMMs + softmax + o_proj reduction. Bump bf16 tolerance slightly.
    scale = 4.0 if dtype == torch.bfloat16 else 1.0
    torch.testing.assert_close(y, y_ref, **tol(dtype, base=scale))


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_tp_gqa(tp_size: int, dtype_str: str) -> None:
    require_gpus(tp_size)
    run_on_ranks(tp_size, _gqa_worker, dtype_str)
