"""Exercise 8 — Wei's hybrid intra-lean/inter-dispatch schedule.

**This is our new idea to explore.** Under TP × DP × EP case (3) topology,
attention TP-4 produces hidden states replicated within each TP group.
`ex06_ep_pure` and `ex07`'s canonical schedule redundantly dispatch
intra-TP-group tokens even though the destination rank already has
them via TP replication.

**The hybrid schedule fixes this**:
- **INTRA-TP records** (dest expert on a rank in the same TP group):
  every rank in the TP group filters to its OWN local experts, computes
  locally, contributes to a shared partial_output. Within-TP `all_reduce`
  sums across TP-group ranks (each contributed to disjoint experts).
- **INTER-TP records** (dest expert in a different TP group): stripe
  within the TP group (rank r takes `[r::tp_size]`), dispatch across
  EP group using `all_to_all_variable` with **zero counts for intra-group
  destinations** (they're already handled above). Combine and scatter
  into the same partial_output.
- **Single `all_reduce` within TP group** distributes both intra and
  inter contributions.

**Bandwidth**: ~37% less than Ex07's canonical schedule under uniform
routing (case 3 config). At skewed routing, intra work stays balanced
via within-TP `all_reduce` regardless of skew.

Read [README.md](README.md) for the two-branch schedule diagram, the
zero-count all_to_all_v trick, and verification claims.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex08_tp_ep_weiz.py -v
```
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex00_dist_primer.solution import all_to_all_variable
from bootcamp.ex04_gqa_tp.reference import TPGQA
from bootcamp.ref.block import RMSNorm
from bootcamp.ref.mlp import RefSwiGLU_MLP


class HybridScheduleMoE(nn.Module):
    """MoE with intra-TP-lean + inter-TP-dispatch hybrid schedule.

    Constructor args identical to Ex07's HybridMoE — same TP + EP parameters.
    Only forward's collective schedule differs.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        ep_group: dist.ProcessGroup | None,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        # TODO(you): same as Ex07 HybridMoE's __init__.
        # 1. assert num_experts % ep_size == 0.
        # 2. Store hyperparams (hidden, intermediate, num_experts, top_k,
        #    tp_size, tp_rank, tp_group, ep_size, ep_rank, ep_group, norm_topk_prob).
        # 3. self.experts_per_rank / expert_start / expert_end.
        # 4. self.my_tp_group_idx = ep_rank // tp_size — which TP group I'm in.
        # 5. self.gate = nn.Linear(hidden, num_experts, bias=False) — REPLICATED.
        # 6. self.experts = nn.ModuleList of experts_per_rank RefSwiGLU_MLPs — SHARDED.
        raise NotImplementedError

    def weight_loader(
        self,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        # TODO(you): same as Ex07 HybridMoE's weight_loader.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden)
        N_tp = x_flat.shape[0]

        # =====================================================================
        # Phase 1: local router (redundant across TP group but cheap).
        # =====================================================================
        # TODO(you):
        # - router_logits = self.gate(x_flat)
        # - top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        # - softmax + optional renorm; cast weights to x.dtype.
        # - top_k_experts_flat = top_k_experts.reshape(-1)   [N_tp * k]
        # - top_k_weights_flat = top_k_weights.reshape(-1)   [N_tp * k]
        # - token_ids_flat = arange(N_tp, device=x.device).repeat_interleave(self.top_k)

        # =====================================================================
        # Phase 2: classify each record as INTRA-TP or INTER-TP.
        # =====================================================================
        # TODO(you):
        # - dest_ranks       = top_k_experts_flat // self.experts_per_rank  # [N_tp * k]
        # - dest_tp_groups   = dest_ranks // self.tp_size
        # - intra_mask       = (dest_ranks == self.ep_rank)  # this rank's OWN records only
        # - inter_mask       = (dest_tp_groups != self.my_tp_group_idx)
        # NOTE: records with (intra_dest_tp_group AND dest_rank != my_rank) are handled
        #       by OTHER ranks in my TP group; the final TP all_reduce sums their contributions.

        partial_output = torch.zeros(N_tp, self.hidden, device=x.device, dtype=x_flat.dtype)

        # =====================================================================
        # Phase 3: INTRA branch — filter to (dest == my_rank), local compute, scatter.
        # No dispatch, no cross-network transport.
        # =====================================================================
        # TODO(you):
        # - intra_positions = intra_mask.nonzero(as_tuple=True)[0]
        # - intra_token_ids = token_ids_flat[intra_positions]
        # - intra_expert_ids = top_k_experts_flat[intra_positions]
        # - intra_weights = top_k_weights_flat[intra_positions]
        # - Sort by local expert (expert_id - self.expert_start), stable=True.
        # - Local per-expert compute loop (identical to Ex06 Phase 7).
        # - Multiply by weights, scatter into partial_output via index_add_.

        # =====================================================================
        # Phase 4: INTER branch — stripe within TP, dispatch across EP, combine.
        # =====================================================================
        # TODO(you):
        # - inter_positions_all = inter_mask.nonzero(as_tuple=True)[0]
        # - my_stripe = inter_positions_all[self.tp_rank :: self.tp_size]  # rank r's 1/tp_size share
        # - stripe_token_ids  = token_ids_flat[my_stripe]
        # - stripe_expert_ids = top_k_experts_flat[my_stripe]
        # - stripe_weights    = top_k_weights_flat[my_stripe]
        # - Sort by dest expert_id, gather sorted_x, compute dest_ranks.
        # - input_split_sizes = bincount(stripe_dest_ranks, minlength=self.ep_size)
        #   NOTE: intra destinations get ZERO count (no records go to intra dests
        #         in the inter branch — that's the "zero-count trick").
        # - Negotiate output_split_sizes via all_to_all_single over EP group.
        # - Dispatch via all_to_all_variable × 2 (x and expert_ids) over EP group.
        # - Local re-argsort by local expert_id + compute (identical to Ex06 Phase 6-7).
        # - Reverse local sort → unsorted_received_out.
        # - Combine via all_to_all_variable (swap splits) over EP group.
        # - Multiply returned_out by stripe_sorted_weights[:, None] (weights stayed local).
        # - partial_output.index_add_(0, stripe_sorted_token_ids, returned_weighted).

        # =====================================================================
        # Phase 5: all_reduce within TP group — sums intra + inter contributions.
        # =====================================================================
        # TODO(you):
        # - dist.all_reduce(partial_output, op=dist.ReduceOp.SUM, group=self.tp_group)

        # =====================================================================
        # Return.
        # =====================================================================
        # TODO(you):
        # - return partial_output.reshape(original_shape)

        raise NotImplementedError


class HybridScheduleBlock(nn.Module):
    """One transformer block using Wei's hybrid schedule MoE.

    Attention path identical to Ex07's HybridBlock — TP-4 within tp_group.
    Only the MoE differs.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        num_experts: int,
        top_k: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        ep_group: dist.ProcessGroup | None,
        norm_topk_prob: bool = True,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # TODO(you): same as Ex07 HybridBlock's __init__, but with HybridScheduleMoE.
        raise NotImplementedError

    def weight_loader(
        self,
        attn_norm_weight: torch.Tensor,
        q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        moe_norm_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        # TODO(you): same as Ex07 HybridBlock's weight_loader.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): same as Ex07 HybridBlock's forward.
        # - h = x + self.attn(self.attn_norm(x))
        # - y = h + self.moe(self.moe_norm(h))
        # - return y
        raise NotImplementedError
