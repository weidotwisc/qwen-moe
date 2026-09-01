"""Exercise 10 — Fused hybrid: Ex09 fused MoE kernel inside Ex07 HybridBlock.

The composition target of the paper. Two previously-verified components
(Ex07's `HybridBlock` + `HybridMoE`; Ex09's `fused_moe_forward`) snapped
together at a single call boundary.

## What you fill in

Subclass Ex07's `HybridMoE` and override:

    __init__: pre-pack the nn.ModuleList of expert weights into
              contiguous [E_per_rank, I, H] / [E_per_rank, H, I] tensors
              using `pack_expert_weights` from ex09's reference. Keep
              the nn.ModuleList too so weight_loader still works.

    forward: identical to Ex07's up through the local sort-by-expert
             step (steps 1-5, ending with x_local_T_sorted and
             expert_ids_local_T_sorted). Then swap the Python expert
             loop (Ex07 lines ~205-211) for a single call:

                 offsets = F.pad(counts.cumsum(0), (1, 0)).to(torch.int64)
                 y_local_T_sorted = fused_moe_forward(
                     x_local_T_sorted, offsets,
                     self.W_gate_packed, self.W_up_packed, self.W_down_packed,
                 )

             Everything after (steps 6-9: reverse-sort, all-to-all back,
             TP gather, weight and combine, allreduce) stays identical
             to Ex07.

The `FusedHybridBlock` wrapper mirrors Ex07's `HybridBlock` structure
with `FusedHybridMoE` swapped in for `HybridMoE`.

## Run

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest \\
        bootcamp/tests/test_ex10_fused_moe_hybrid.py -v
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bootcamp.ex07_tp_ep_hybrid.solution import HybridBlock, HybridMoE
from bootcamp.ex09_fused_moe.reference import (
    fused_moe_forward,
    pack_expert_weights,
)


class FusedHybridMoE(HybridMoE):
    """HybridMoE with the local per-expert Python loop replaced by a single
    call to Ex09's fused_moe_forward.

    Precondition (identical to HybridMoE): all attributes set by
    HybridMoE.__init__ are populated — self.experts (nn.ModuleList of
    RefSwiGLU_MLP), self.experts_per_rank, self.expert_start,
    self.ep_group, self.tp_group, self.router, self.top_k,
    self.norm_topk_prob.

    Postcondition: forward(x) returns the same tensor as HybridMoE.forward(x)
    up to declared floating-point reduction-order tolerance
    (fp32: atol=rtol=1e-4; bf16: atol=rtol=5e-2).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wei to implement:
        #   Call pack_expert_weights(self.experts) once, register the
        #   resulting [E_per_rank, I, H] / [E_per_rank, H, I] tensors as
        #   buffers (they mirror self.experts, so no new params).
        raise NotImplementedError("Ex10.__init__ pending — pack weights here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Wei to implement:
        #   Copy HybridMoE.forward steps 1-5 verbatim (routing, all-to-all
        #   dispatch, sort by expert id), replace the expert loop (steps
        #   ~205-211) with fused_moe_forward, keep steps 6-9 verbatim
        #   (unsort, all-to-all back, TP gather, weight & combine, allreduce).
        raise NotImplementedError("Ex10.forward pending")


class FusedHybridBlock(HybridBlock):
    """HybridBlock with FusedHybridMoE swapped in for HybridMoE.

    Same TP-4 × DP-2 × EP-8 (or TP-8 × DP-2 × EP-16 for 2-node) topology;
    same attention path; only the MoE local-compute step changes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wei to implement:
        #   Replace self.moe: HybridMoE with an equivalent FusedHybridMoE
        #   (constructed with the same args used by super().__init__).
        raise NotImplementedError("Ex10 block wiring pending")
