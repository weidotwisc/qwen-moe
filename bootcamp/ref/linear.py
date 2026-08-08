"""Trivial single-GPU reference linear.

Present for API symmetry with the other reference modules — every exercise's
test compares against a `Ref*` class. For L1 that Ref is basically nn.Linear.
"""

from __future__ import annotations

import torch
from torch import nn


class RefLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
