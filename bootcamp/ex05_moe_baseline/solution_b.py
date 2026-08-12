"""Exercise 5b — Permuted MoE with grouped compute (single-GPU).

Same MoE math as Ex05a, but with a different **data layout**: tokens
are permuted so each expert's inputs are contiguous. Each expert then
runs one grouped matmul on its contiguous slice, instead of scattered
per-token accesses.

## Why this exercise matters

This is the algorithmic bridge between Ex05a and everything after.

**Ex06 (Expert Parallelism)** is essentially this exercise + an
`all_to_all_variable` inserted between the permutation and the
grouped compute. The permutation vocabulary (argsort by expert,
bincount for offsets, unpermute at the end) is exactly what nanovllm's
`_forward_expert_parallel` uses.

**Ex08 (Fused MoE Triton kernel)** is this same algorithm, but with
the Python for-loop replaced by a single Triton kernel launch that
does the grouped matmul. Same permutation vocabulary; same expert
offsets; same unpermute.

If you get comfortable with this permutation dance now, Ex06 and Ex08
feel like small deltas rather than fresh problems.

## The math

Given input `x` of shape `[N, hidden]`, router weights (from Ex05a's
gate + softmax + top-k) `routing_weights` of shape `[N, top_k]` and
`selected_experts` of shape `[N, top_k]`:

1. **Flatten routing to (N * top_k) triples**: `(token_id, expert_id, weight)`.
2. **Argsort by expert_id** to get a permutation. After permuting,
   tokens for expert 0 come first, then expert 1, ..., then expert E-1.
3. **Compute expert offsets** via `bincount(sorted_expert_ids)` +
   `cumsum` → gives per-expert start/end positions.
4. **Gather** the input `x[sorted_token_ids]` into a contiguous
   `sorted_x` of shape `[N * top_k, hidden]`.
5. **Grouped compute**: for each expert e, run `experts[e](sorted_x[offsets[e]:offsets[e+1]])`
   on the CONTIGUOUS slice. One matmul per expert with many tokens,
   instead of many tiny matmuls.
6. **Weight and unpermute**: multiply outputs by `sorted_weights[:, None]`,
   then `index_add_` back into `output` at the original `sorted_token_ids`
   positions (each token's contributions from its top_k experts sum up).

## What to fill in

Fill in `PermutedSparseMoE`. The router (steps 1-4 of Ex05a) is the
same; only the per-expert compute + combine changes.

## Numerical tolerance vs Ex05a

The output is the **same math** as Ex05a's per-expert loop, but the
accumulation order differs (permuted vs by-expert). fp arithmetic
isn't strictly associative, so exact byte equivalence isn't
guaranteed — expect ~1e-5 differences in fp32, ~1e-2 in bf16. Tests
use `assert_close` with generous atol/rtol accordingly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


class PermutedSparseMoE(nn.Module):
    """Single-GPU sparse MoE with permutation + grouped compute.

    Args identical to `NaiveSparseMoE` — same abstract module, different
    implementation.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        # TODO(you): same as Ex05a's __init__. Store hyperparams and allocate
        # self.gate (Linear) + self.experts (ModuleList of RefSwiGLU_MLP).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permuted forward.

        Args:
            x: [B, T, H] or [N, H] input.
        Returns:
            Tensor of the same shape as x.
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        N = x_flat.shape[0]

        # ============ Step 1: Router (SAME as Ex05a) ============
        # TODO(you):
        # - router_logits = self.gate(x_flat)                               # [N, E]
        # - routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # - routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        # - if norm_topk_prob: normalize routing_weights.
        # - Cast routing_weights back to x_flat.dtype.
        # After this step:
        #   selected_experts: [N, top_k]  expert IDs per token per k
        #   routing_weights:  [N, top_k]  weights per token per k (sum to 1 per token if norm)

        # ============ Step 2: Flatten routing to (N * top_k) triples ============
        # Every (token, expert-choice) pair becomes an independent record.
        # TODO(you):
        # - token_ids  = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        #   equivalently: torch.arange(N).unsqueeze(-1).expand(N, top_k).reshape(-1)
        #   Result: [0, 0, ..., 0, 1, 1, ..., 1, ..., N-1, ..., N-1]  (each rank 0..N-1 repeated top_k times)
        # - expert_ids = selected_experts.reshape(-1)                         # [N * top_k]
        # - weights    = routing_weights.reshape(-1)                          # [N * top_k]

        # ============ Step 3: Argsort by expert_id to compute permutation ============
        # TODO(you):
        # - permutation = torch.argsort(expert_ids)                           # [N * top_k]
        #   After permuting, all records for expert 0 come first, then expert 1, ...
        # - sorted_expert_ids = expert_ids[permutation]
        # - sorted_token_ids  = token_ids[permutation]
        # - sorted_weights    = weights[permutation]

        # ============ Step 4: Compute per-expert offsets ============
        # TODO(you):
        # - counts  = torch.bincount(sorted_expert_ids, minlength=self.num_experts)   # [num_experts]
        # - offsets = torch.cat([
        #       torch.zeros(1, dtype=torch.long, device=x.device),
        #       counts.cumsum(0),
        #   ])   # [num_experts + 1]. offsets[e] is where expert e's slice starts.

        # ============ Step 5: Gather sorted input ============
        # TODO(you):
        # - sorted_x = x_flat[sorted_token_ids]  # [N * top_k, hidden]

        # ============ Step 6: Grouped per-expert compute ============
        # TODO(you):
        # - sorted_out = torch.empty_like(sorted_x)
        # - for e in range(self.num_experts):
        #       start, end = int(offsets[e]), int(offsets[e + 1])
        #       if start == end: continue
        #       sorted_out[start:end] = self.experts[e](sorted_x[start:end])
        #   NOTE: unlike Ex05a's naive loop, each expert here gets a CONTIGUOUS slice —
        #   the underlying matmul is one big call, not many small ones per token.

        # ============ Step 7: Weight + unpermute + combine ============
        # TODO(you):
        # - sorted_out = sorted_out * sorted_weights.unsqueeze(-1)             # elementwise weight
        # - output = torch.zeros_like(x_flat)
        # - output.index_add_(0, sorted_token_ids, sorted_out)
        #   Each token gets top_k contributions summed into its output slot.
        # - return output.reshape(original_shape)
        raise NotImplementedError
