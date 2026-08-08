"""Shared pytest fixtures + constants for the TP/EP bootcamp tests."""

from __future__ import annotations

import torch

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def tol(dtype: torch.dtype, *, base: float = 1.0) -> dict[str, float]:
    """Correctness tolerance keyed by dtype. `base` scales up for exercises
    that accumulate more (e.g. MoE combine, sum over 8 experts).

    fp32: 1e-5 baseline; bf16: 5e-2 baseline (bf16 has ~7 bits of mantissa,
    so a couple percent error over a few-hundred-length dot product is normal).
    """
    if dtype == torch.float32:
        return {"atol": 1e-5 * base, "rtol": 1e-5 * base}
    if dtype == torch.bfloat16:
        return {"atol": 5e-2 * base, "rtol": 5e-2 * base}
    raise ValueError(f"unsupported dtype {dtype}")
