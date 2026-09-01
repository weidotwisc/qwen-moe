"""Reference implementation for Ex10 — Fused hybrid: Ex09 kernel inside Ex07.

Working manual composition of two prior microbenchmarks:
- Ex07's `HybridBlock` / `HybridMoE`: TP-4 × DP-2 × EP-8 topology
  (or TP-8 × DP-2 × EP-16 for 2-node), routing, all-to-all dispatch/
  combine, TP gather.
- Ex09's `fused_moe_forward`: single Triton grouped-GEMM launch that
  replaces the per-expert Python loop.

The composition boundary is intentionally minimal: exactly one method
of one class changes (`HybridMoE.forward`'s expert-loop stanza,
Ex07 solution lines ~205-211). Everything else — the routing math,
the dispatch collectives, the TP gather, the residual wiring — is
inherited unchanged from Ex07.

Contract (identical to Ex07's `HybridBlock`):
    Input:  x on rank r = [B, T, H], replicated within tp_group.
    Output: y on rank r = [B, T, H], replicated within tp_group.

The composition proof reduces to chaining Ex07's postcondition on
`(x_local_T_sorted, expert_ids_local_T_sorted)` with Ex09's
precondition on `(sorted_x, offsets)`. Both are shape + routing-
conservation contracts; see the paper's composition theorem.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex07_tp_ep_hybrid.solution import HybridBlock, HybridMoE
from bootcamp.ex09_fused_moe.reference import (
    fused_moe_forward,
    pack_expert_weights,
)


class FusedHybridMoE(HybridMoE):
    """HybridMoE with the local per-expert Python loop replaced by one
    call into Ex09's fused_moe_forward.

    Weights are packed lazily on the first forward pass, after
    `weight_loader` has run. The packed [E_per_rank, I, H] and
    [E_per_rank, H, I] tensors are cached on the instance for
    subsequent forwards.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Populated on first forward (weight_loader must run first).
        self._packed: dict[str, torch.Tensor] | None = None

    def _ensure_packed(self) -> dict[str, torch.Tensor]:
        if self._packed is None:
            self._packed = pack_expert_weights(self.experts)
        return self._packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """See Ex07 HybridMoE.forward for the full step-by-step; only the
        step-5 expert loop is replaced here.
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden)
        N_tp = x_flat.shape[0]
        assert N_tp % self.tp_size == 0
        local_N = N_tp // self.tp_size

        # Phase 0: TP-group stripe.
        local_x = x_flat[self.tp_rank * local_N : (self.tp_rank + 1) * local_N]
        local_y = torch.zeros_like(local_x)
        y_flat = torch.zeros_like(x_flat)

        # Phase 1: router (local).
        router_logits = self.gate(local_x)
        router_logits_top_k, top_k_expert_ids = torch.topk(
            router_logits, k=self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            weights_top_k = F.softmax(router_logits_top_k, dim=-1)
        else:
            weights_top_k = F.softmax(router_logits, dim=-1).gather(
                dim=-1, index=top_k_expert_ids
            )
        weights_top_k = weights_top_k.to(local_x.dtype)

        # Phase 2: sort by (global) expert id.
        top_k_expert_ids_flat = top_k_expert_ids.reshape(-1)
        weights_top_k_flat = weights_top_k.reshape(-1)
        top_k_expert_ids_flat_sorted, top_k_expert_ids_perm = torch.sort(
            top_k_expert_ids_flat, stable=True
        )
        top_k_expert_ids_perm_normalized = top_k_expert_ids_perm // self.top_k
        x_repeated_k = local_x[top_k_expert_ids_perm_normalized]
        y_repeated_k_sorted = torch.zeros_like(x_repeated_k)
        y_repeated_k = torch.zeros_like(x_repeated_k)

        # Phase 3: negotiate send/recv counts over EP.
        top_k_expert_ids_flat_sorted_input_cnts = torch.bincount(
            top_k_expert_ids_flat_sorted // self.experts_per_rank,
            minlength=self.ep_size,
        )
        top_k_expert_ids_flat_sorted_output_cnts = torch.zeros_like(
            top_k_expert_ids_flat_sorted_input_cnts
        )
        dist.all_to_all_single(
            top_k_expert_ids_flat_sorted_output_cnts,
            top_k_expert_ids_flat_sorted_input_cnts,
            group=self.ep_group,
        )

        # Phase 4: dispatch tokens + their expert ids over EP.
        input_cnts_list = top_k_expert_ids_flat_sorted_input_cnts.tolist()
        output_cnts_list = top_k_expert_ids_flat_sorted_output_cnts.tolist()
        x_local_T = torch.zeros(
            size=(sum(output_cnts_list), original_shape[-1]),
            dtype=x.dtype, device=x.device,
        )
        y_local_T = torch.zeros_like(x_local_T)
        expert_ids_local_T = torch.zeros(
            size=(sum(output_cnts_list),), dtype=torch.long, device=x.device
        )
        dist.all_to_all_single(
            x_local_T, x_repeated_k, output_cnts_list, input_cnts_list,
            group=self.ep_group,
        )
        dist.all_to_all_single(
            expert_ids_local_T, top_k_expert_ids_flat_sorted,
            output_cnts_list, input_cnts_list, group=self.ep_group,
        )
        expert_ids_local_T -= self.expert_start  # rebase to local expert ids

        # Phase 5: sort locally by expert id, then FUSED expert compute.
        expert_ids_local_T_sorted, expert_ids_local_T_perm = torch.sort(
            expert_ids_local_T, stable=True
        )
        x_local_T_sorted = x_local_T[expert_ids_local_T_perm]

        # ---- THE ONE-LINE COMPOSITION ----------------------------------
        # Ex07 does this via a Python loop over self.experts (lines
        # ~205-211 of ex07_tp_ep_hybrid/solution.py); Ex10 replaces the
        # loop with a single call into Ex09's fused_moe_forward.
        counts = torch.bincount(
            expert_ids_local_T_sorted, minlength=self.experts_per_rank
        )
        offsets = F.pad(counts.cumsum(0), (1, 0)).to(torch.int64)
        packed = self._ensure_packed()
        y_local_T_sorted = fused_moe_forward(
            x_local_T_sorted, offsets,
            packed["W_gate"], packed["W_up"], packed["W_down"],
        )
        # ----------------------------------------------------------------

        y_local_T[expert_ids_local_T_perm] = y_local_T_sorted  # unsort

        # Phase 6: combine — reverse-dispatch outputs back over EP.
        dist.all_to_all_single(
            y_repeated_k_sorted, y_local_T,
            output_split_sizes=input_cnts_list,
            input_split_sizes=output_cnts_list,
            group=self.ep_group,
        )

        # Phase 7: unsort by top-k perm, weight, scatter-add back to tokens.
        y_repeated_k[top_k_expert_ids_perm] = y_repeated_k_sorted
        y_repeated_k *= weights_top_k_flat[:, None]
        local_y.index_add_(
            dim=0,
            index=torch.arange(local_N, device=x.device).repeat_interleave(
                repeats=self.top_k, dim=-1
            ),
            source=y_repeated_k,
        )

        # Phase 8: TP gather to honor "replicated within tp_group" invariant.
        dist.all_gather_into_tensor(y_flat, local_y, group=self.tp_group)

        # Phase 9: reshape to caller's [B, T, H].
        return y_flat.reshape(original_shape)


class FusedHybridBlock(HybridBlock):
    """Ex07's HybridBlock with FusedHybridMoE swapped in for HybridMoE.

    Same topology, same attention path, same wiring; only the MoE
    submodule differs. The weight_loader inherited from HybridBlock
    still populates `self.moe.experts` correctly because
    FusedHybridMoE.experts is inherited from HybridMoE.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Replace HybridMoE instance with FusedHybridMoE, using the same
        # constructor args super() used. Read them off self.moe.
        moe = self.moe
        assert isinstance(moe, HybridMoE)
        self.moe = FusedHybridMoE(
            hidden=moe.hidden,
            intermediate=moe.intermediate,
            num_experts=moe.num_experts,
            top_k=moe.top_k,
            tp_size=moe.tp_size,
            tp_rank=moe.tp_rank,
            tp_group=moe.tp_group,
            ep_size=moe.ep_size,
            ep_rank=moe.ep_rank,
            ep_group=moe.ep_group,
            norm_topk_prob=moe.norm_topk_prob,
        ).to(device=moe.gate.weight.device, dtype=moe.gate.weight.dtype)
