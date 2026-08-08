"""Single-GPU SwiGLU MLP (Qwen3 expert shape).

    y = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RefSwiGLU_MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
