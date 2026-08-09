"""Exercise 2 — SwiGLU MLP under TP.

Two things to fill in:

1. `MergedColumnParallelLinear` — a subclass of ex01's ColumnParallelLinear
   that fuses TWO output projections (gate + up) into a single weight matrix
   so we only pay for one shard-storage / one gemm launch per forward.

2. `TPSwiGLUMLP` — a full SwiGLU MLP block: MergedColumnParallelLinear for the
   fused (gate, up) projection → SiLU-and-multiply → RowParallelLinear for
   the down projection.

Depends on your ex01 solution.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex01_linear_tp.solution import ColumnParallelLinear, RowParallelLinear


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Multiple linear projections that share an input and can be fused into one
    weight matrix. Each of the `output_sizes` projections is independently
    TP-sharded on the out-dim; the shards are concatenated in the merged weight.

    Example (gate + up, each with `intermediate` outputs, under tp_size=4):
        full weight shape:      [2 * intermediate, in_features]
        this-rank weight shape: [2 * intermediate / 4, in_features]
                                  = [gate_shard_rank | up_shard_rank] on dim 0.

    Weight loading is per-projection: call `weight_loader(gate_full_weight, 0)`
    then `weight_loader(up_full_weight, 1)`.
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        # Reuse ColumnParallelLinear's __init__ with the total output.
        super().__init__(in_features, sum(output_sizes), tp_size, tp_rank, group=group)
        self.output_sizes = output_sizes

    def weight_loader(self, full_weight: torch.Tensor, shard_id: int) -> None:  # type: ignore[override]
        """Copy this rank's slice of the shard_id-th projection into the correct
        position in self.weight.

        full_weight: [output_sizes[shard_id], in_features]  — one projection's
                     un-sharded weight.
        shard_id:    which projection this is (0 = gate, 1 = up in the SwiGLU
                     usage below).
        """
        # TODO(you):
        # 1. Compute the offset within self.weight (dim 0) where projection
        #    `shard_id` starts. Hint: sum of the per-rank sizes of all earlier
        #    projections.
        # 2. Compute this rank's slice of `full_weight` on dim 0 (same math as
        #    ex01's ColumnParallelLinear.weight_loader).
        # 3. Copy the slice into self.weight.data.narrow(0, offset, shard_size).
        raise NotImplementedError


class TPSwiGLUMLP(nn.Module):
    """A full SwiGLU MLP block, TP-sharded on the intermediate dim.

        gate_up = MergedColumnParallelLinear(hidden, [intermediate, intermediate])
        y       = SiLU(gate) * up                              # elementwise on the shard
        out     = RowParallelLinear(intermediate, hidden)(y)   # all_reduce inside

    Because the SiLU-and-multiply is elementwise and each rank holds a
    matching slice of gate and up, no cross-rank comm is needed until the
    RowParallelLinear all_reduce at the end. Two collectives budget per block
    would be one too many; this design has EXACTLY ONE.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        # TODO(you): allocate self.gate_up_proj (MergedColumnParallelLinear)
        # and self.down_proj (RowParallelLinear). Pass `group=group` down to both.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you):
        # 1. gate_up = self.gate_up_proj(x)   # [..., 2 * intermediate / tp_size]
        # 2. split into gate, up on the last dim.
        # 3. hidden = F.silu(gate) * up       # [..., intermediate / tp_size]
        # 4. return self.down_proj(hidden)    # replicated [..., hidden]
        raise NotImplementedError
