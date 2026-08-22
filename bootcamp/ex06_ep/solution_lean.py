"""Exercise 6 — Expert Parallelism, LEAN VARIANT (single all_reduce).

Fill in `EPSparseMoE`. Same MoE semantics as `solution.py`, but exploits
the replicated-input pre-condition to skip the dispatch collective
entirely. **One all_reduce per forward** replaces the four-collective
dispatch/combine/gather schedule.

## Contrast with solution.py (dispatch-based)

| Aspect | solution.py (dispatch) | solution_lean.py (this) |
|---|---|---|
| Collective count | 4 (splits + 2 all_to_all_v + all_gather) | **1 (all_reduce)** |
| Bytes at Qwen3 config | ~4·N·H | ~1.75·N·H |
| Load balance under skew | data-dependent | data-independent |
| Compute-comm pipelining | possible (fused kernels) | possible (streaming reduce) |
| Handles non-replicated input | yes | no — requires replication |
| Scales past ep = top_k + 1 | yes (message shrinks) | no (constant transfer) |

Both variants produce identical output up to fp reduction-order noise.
Both maintain: `[N, H] replicated on ep_group → forward → [N, H] replicated on ep_group`.

## The 7-phase pipeline

Read [README.md](README.md) for design rationale. Summary of phases:

1. **Full router on all N tokens** (redundant across ranks — cheap).
2. **Filter routing records to LOCAL experts**.
3. **Local argsort by local expert_id + gather sorted_x LOCALLY**.
4. **bincount + cumsum → local offsets**.
5. **Local expert compute** on contiguous slices.
6. **Build partial_output [N, H]** with zeros for non-contributed tokens.
7. **ONE all_reduce** — assembles full output on every rank.

**No dispatch. No combine. No final all_gather.** Every rank contributes
`[N, H]` to the all_reduce; the sum recovers each token's full
weighted-expert contribution.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex06_ep_lean.py -v
```
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


class EPSparseMoE(nn.Module):
    """Expert-parallel sparse MoE — lean (all_reduce) variant.

    Args:
        hidden: input/output dim.
        intermediate: per-expert MLP intermediate dim.
        num_experts: total experts across all ranks (must be divisible by ep_size).
        top_k: experts per token.
        ep_size: number of EP ranks.
        ep_rank: this rank's id in [0, ep_size).
        group: EP process group. None means the default world group.
        norm_topk_prob: if True, renormalize top-k weights to sum to 1.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        group: dist.ProcessGroup | None = None,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        # TODO(you):
        # Same as solution.py's __init__ — the class shape is identical.
        # 1. assert num_experts % ep_size == 0
        # 2. Store hyperparams (hidden, intermediate, num_experts, top_k,
        #    ep_size, ep_rank, group, norm_topk_prob).
        # 3. self.experts_per_rank = num_experts // ep_size
        #    self.expert_start     = ep_rank * self.experts_per_rank
        #    self.expert_end       = self.expert_start + self.experts_per_rank
        # 4. self.gate = nn.Linear(hidden, num_experts, bias=False)  — REPLICATED
        # 5. self.experts = nn.ModuleList of experts_per_rank RefSwiGLU_MLPs — SHARDED
        raise NotImplementedError

    def weight_loader(
        self,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        """Load full weights into this rank's local shard.

        Args identical to solution.py's weight_loader. Copies:
        - gate_weight fully (replicated).
        - Only [expert_start, expert_end) of the expert weight lists.
        """
        # TODO(you): same as solution.py's weight_loader.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lean EP forward.

        Args:
            x: [B, T, H] or [N, H], REPLICATED across ep_group.
        Returns:
            [B, T, H] or [N, H], REPLICATED (assembled via a single all_reduce).
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        N = x_flat.shape[0]

        # =====================================================================
        # Phase 1: FULL router on all N tokens.
        # Every rank sees identical routing (deterministic on replicated input).
        # =====================================================================
        # TODO(you):
        # - router_logits = self.gate(x_flat)                                  # [N, num_experts]
        # - top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        # - if self.norm_topk_prob:
        #       top_k_weights = F.softmax(top_k_weights, dim=-1)
        #   else:
        #       top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        # - top_k_weights = top_k_weights.to(x_flat.dtype)

        # =====================================================================
        # Phase 2: filter routing records to LOCAL experts.
        # Each rank keeps only records where the destination expert is in its range.
        # =====================================================================
        # TODO(you):
        # - top_k_experts_flat = top_k_experts.reshape(-1)                     # [N*k]
        # - top_k_weights_flat = top_k_weights.reshape(-1)                     # [N*k]
        # - token_ids_flat     = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        # - local_mask = (top_k_experts_flat >= self.expert_start) & \
        #                (top_k_experts_flat <  self.expert_end)
        # - local_idx  = local_mask.nonzero(as_tuple=True)[0]                  # [num_local_records]
        # - local_expert_ids = top_k_experts_flat[local_idx] - self.expert_start
        # - local_weights    = top_k_weights_flat[local_idx]
        # - local_token_ids  = token_ids_flat[local_idx]

        # =====================================================================
        # Phase 3: local argsort by LOCAL expert_id.
        # No collective — everything is on-rank.
        # =====================================================================
        # TODO(you):
        # - sorted_local_ids, sort_perm = torch.sort(local_expert_ids, stable=True)
        # - sorted_weights   = local_weights[sort_perm]
        # - sorted_token_ids = local_token_ids[sort_perm]
        # - sorted_x         = x_flat[sorted_token_ids]                        # LOCAL fancy-index

        # =====================================================================
        # Phase 4: bincount + offsets.
        # =====================================================================
        # TODO(you):
        # - local_counts  = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
        # - local_offsets = F.pad(local_counts.cumsum(0), (1, 0))              # [experts_per_rank + 1]

        # =====================================================================
        # Phase 5: local expert compute (same as Ex05b).
        # =====================================================================
        # TODO(you):
        # - expert_out = torch.empty_like(sorted_x)
        # - for local_e in range(self.experts_per_rank):
        #       s = local_offsets[local_e].item()
        #       e = local_offsets[local_e + 1].item()
        #       if s == e:
        #           continue
        #       expert_out[s:e] = self.experts[local_e](sorted_x[s:e])
        # - expert_out = expert_out * sorted_weights[:, None]                  # weight per record

        # =====================================================================
        # Phase 6: build partial_output [N, H].
        # Zeros for tokens with no local-expert contribution;
        # local scatter-add for tokens with at least one local contribution.
        # =====================================================================
        # TODO(you):
        # - partial_output = torch.zeros(N, self.hidden, device=x.device, dtype=x.dtype)
        # - partial_output.index_add_(0, sorted_token_ids, expert_out)

        # =====================================================================
        # Phase 7: ONE all_reduce — assembles full replicated output.
        # In-place. Payload: N·H bytes per rank per direction.
        # Data-independent — identical bytes regardless of routing distribution.
        # =====================================================================
        # TODO(you):
        # - dist.all_reduce(partial_output, op=dist.ReduceOp.SUM, group=self.group)
        # - return partial_output.reshape(original_shape)

        raise NotImplementedError
