"""Exercise 6 — Expert Parallelism (EP) with all-to-all dispatch/combine.

Fill in `EPSparseMoE`. The core algorithm is Ex05b's permuted MoE, with
**two `all_to_all_variable` collectives inserted around the local expert
loop** — one for dispatch (send tokens to their expert-owning ranks),
one for combine (send outputs back).

## Design

- **Input**: `x: [B, T, H]` or `[N, H]` — REPLICATED across the EP group.
- **Token partition**: rank r owns tokens `[r * (N/ep_size), (r+1) * (N/ep_size))`
  (contiguous chunk). No striping in Ex06 — Ex07 introduces striping.
- **Expert partition**: rank r owns experts `[r * experts_per_rank, (r+1) * experts_per_rank)`.
- **Output**: `[N, H]` REPLICATED — reconstructed via `all_gather` at end.

## Design choice: weights never cross the network

The dispatch payload carries `(x, expert_ids)` — NOT the routing weights
or the source-token indices. Those stay on the originating rank. The
compute rank sees pure `expert(x) → x` — no weight-multiply. The final
weight-multiply + scatter happen on the originator after combine.

This matches the paper's abstract framing: experts are pure functions;
routing weights belong to the (token, router) interaction, not the
(token, expert) interaction.

## Read PRE-CONDITIONS

Read [README.md](README.md) for the 11-phase pipeline diagram, the
weight-multiply-on-originator rationale, and traps to watch for.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex06_ep.py -v
```
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex00_dist_primer.solution import all_to_all_variable
#from bootcamp.ref.mlp import RefSwiGLU_MLP

class SwiGLU_MLP(nn.Module):
    def __init__(
        self,
        hidden:int,
        intermediate:int
    ):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=hidden, out_features=intermediate, bias=False) # for MoE FFN, the bias is false
        self.up_proj = nn.Linear(in_features=hidden, out_features=intermediate, bias=False)
        self.down_proj = nn.Linear(in_features=intermediate, out_features=hidden, bias=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class EPSparseMoE(nn.Module):
    """Expert-parallel sparse MoE.

    Each rank owns `num_experts // ep_size` experts and participates in
    two `all_to_all_variable` collectives per forward.

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
        # 1. assert num_experts % ep_size == 0
        # 2. Store hyperparams: hidden, intermediate, num_experts, top_k,
        #    ep_size, ep_rank, group, norm_topk_prob.
        # 3. Compute:
        #      self.experts_per_rank = num_experts // ep_size
        #      self.expert_start     = ep_rank * self.experts_per_rank
        #      self.expert_end       = self.expert_start + self.experts_per_rank
        # 4. self.gate = nn.Linear(hidden, num_experts, bias=False)
        #    — REPLICATED across ranks. Every rank owns the full gate weight.
        # 5. self.experts = nn.ModuleList([
        #        RefSwiGLU_MLP(hidden, intermediate)
        #        for _ in range(self.experts_per_rank)     # ← only OUR experts
        #    ])
        #    — SHARDED across ranks. Rank r only holds experts_per_rank MLPs.

        # weiz step 1: assertion
        assert (num_experts % ep_size == 0)
        # weiz step 2: setters and getters
        self.hidden = hidden
        self.intermediate = intermediate 
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.group = group
        self.norm_topk_prob = norm_topk_prob
        # weiz step 3: compute per rank expers information
        self.experts_per_rank = num_experts // ep_size
        self.expert_start = ep_rank * self.experts_per_rank
        self.expert_end = self.expert_start + self.experts_per_rank
        # weiz step 4: replicate the gate
        self.gate = nn.Linear(hidden, num_experts, bias=False) # remember the bias is False
        # weiz step 5: build my share of experts
        self.experts = nn.ModuleList(
            [SwiGLU_MLP(hidden, intermediate) for _ in range(self.experts_per_rank)]
        )

    def weight_loader(
        self,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        """Load full weights into this rank's local shard.

        Args:
            gate_weight: [num_experts, hidden] — full gate weight (replicated).
            expert_gate_weights: list of [intermediate, hidden], length num_experts.
            expert_up_weights:   list of [intermediate, hidden], length num_experts.
            expert_down_weights: list of [hidden, intermediate], length num_experts.

        Loads gate_weight fully; for experts, only [expert_start, expert_end).
        """
        # TODO(you):
        # 1. Copy gate_weight fully:  self.gate.weight.data.copy_(gate_weight)
        # 2. For each local_e in [0, experts_per_rank):
        #      global_e = self.expert_start + local_e
        #      copy expert_gate_weights[global_e] into self.experts[local_e].gate_proj.weight.data
        #      copy expert_up_weights[global_e]   into self.experts[local_e].up_proj.weight.data
        #      copy expert_down_weights[global_e] into self.experts[local_e].down_proj.weight.data
        
        # weiz step 1 copy gate
        self.gate.weight.data.copy_(gate_weight)
        # weiz step 2
        for local_expert_id in range(self.experts_per_rank):
            global_expert_id = local_expert_id + self.expert_start
            self.experts[local_expert_id].gate_proj.weight.data.copy_(expert_gate_weights[global_expert_id])
            self.experts[local_expert_id].up_proj.weight.data.copy_(expert_up_weights[global_expert_id])
            self.experts[local_expert_id].down_proj.weight.data.copy_(expert_down_weights[global_expert_id])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EP forward.

        Args:
            x: [B, T, H] or [N, H], REPLICATED across the EP group.
        Returns:
            [B, T, H] or [N, H], REPLICATED (reconstructed via all_gather).
        """
        original_shape = x.shape # B, T, H
        x_flat = x.reshape(-1, original_shape[-1]) # N, H
        N = x_flat.shape[0]
        assert N % self.ep_size == 0, (
            f"N={N} must be divisible by ep_size={self.ep_size} for contiguous partition"
        )
        local_N = N // self.ep_size

        # =====================================================================
        # Phase 0: local token partition — this rank owns tokens [local_start, local_end).
        # =====================================================================
        # TODO(you):
        # - local_start = self.ep_rank * local_N
        # - local_end   = local_start + local_N
        # - local_x     = x_flat[local_start:local_end]        # [local_N, H]

        local_start = self.ep_rank * local_N
        local_end = local_start + local_N
        local_x = x_flat[local_start:local_end] # [local_N, H]

        # =====================================================================
        # Phase 1: local router (same math as Ex05b, applied to local_x only).
        # =====================================================================
        # TODO(you): replicate Ex05b's router steps, but on local_x, not x_flat.
        # - router_logits = self.gate(local_x)                                       # [local_N, num_experts]
        # - top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        # - if self.norm_topk_prob:
        #       top_k_weights = F.softmax(top_k_weights, dim=-1)
        #   else:
        #       top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        # - top_k_weights = top_k_weights.to(local_x.dtype)

        router_logits = self.gate(local_x) # [local_N, num_experts]
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1) # [local_N, k]
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(dim=-1,index=top_k_experts)

        # =====================================================================
        # Phase 2: local argsort by GLOBAL expert_id (same vocabulary as Ex05b).
        # =====================================================================
        # TODO(you):
        # - top_k_experts_flat = top_k_experts.reshape(-1)                           # [local_N * k]
        # - top_k_weights_flat = top_k_weights.reshape(-1)                           # [local_N * k]
        # - sorted_expert_ids, sort_perm = torch.sort(top_k_experts_flat, stable=True)
        # - local_token_ids           = torch.arange(local_N, device=x.device).repeat_interleave(self.top_k)
        # - sorted_local_token_ids    = local_token_ids[sort_perm]                   # [local_N * k]
        # - sorted_weights            = top_k_weights_flat[sort_perm]                # [local_N * k]
        # - local_x_permuted          = local_x[sorted_local_token_ids]              # [local_N * k, H]

        top_k_experts_flat = top_k_experts.reshape(-1) # [local_N * k,]
        top_k_weights_flat = top_k_weights.reshape(-1) # [local_N * k,]
        sorted_expert_ids, sort_perm = torch.sort(top_k_experts_flat) # [local_N * k,]
        local_token_ids_rep = torch.arange(local_N, device=x.device).repeat_interleave(dim=-1, repeats=self.top_k) #[local_N*k, ]
        sorted_local_tokens_ids = local_token_ids_rep[sort_perm] # [local_N * k,]
        sorted_weights = top_k_weights_flat[sort_perm] # [local_N * k]
        local_x_permuted = local_x[sorted_local_tokens_ids] # [local_N *k, H]


        # =====================================================================
        # Phase 3: compute per-destination-rank splits.
        # For each record, dest_rank = expert_id // experts_per_rank.
        # =====================================================================
        # TODO(you):
        # - dest_ranks        = sorted_expert_ids // self.experts_per_rank            # [local_N * k]
        # - input_split_sizes = torch.bincount(dest_ranks, minlength=self.ep_size)    # [ep_size], torch.long
        # - Rationale: input_split_sizes[j] = how many records I send to rank j.
        dest_ranks = sorted_expert_ids // self.experts_per_rank # [local_N *k]
        send_buf_sizes = torch.bincount(dest_ranks, minlength=self.ep_size) # [ep_size, ], weiz: 

        # =====================================================================
        # Phase 4: negotiate output_split_sizes via all_gather_into_tensor.
        # Rank j's output_split_sizes[i] must equal rank i's input_split_sizes[j].
        # =====================================================================
        # TODO(you):
        # - all_input_splits = torch.zeros(self.ep_size * self.ep_size,
        #                                   dtype=torch.long, device=x.device)
        # - dist.all_gather_into_tensor(all_input_splits, input_split_sizes, group=self.group)
        # - all_input_splits = all_input_splits.view(self.ep_size, self.ep_size)
        #   # Row i: rank i's input_split_sizes.
        #   # Column j: what rank j receives from each source ⇒ output_split_sizes for rank j.
        # - output_split_sizes = all_input_splits[:, self.ep_rank].contiguous()       # [ep_size]

        # =====================================================================
        # Phase 5: DISPATCH — all_to_all_variable × 2 (x, expert_ids).
        # Weights + sorted_local_token_ids STAY on originator (this rank).
        # =====================================================================
        # TODO(you):
        # - input_splits_list  = input_split_sizes.tolist()
        # - output_splits_list = output_split_sizes.tolist()
        # - received_x           = all_to_all_variable(local_x_permuted,
        #                                              input_splits_list,
        #                                              output_splits_list,
        #                                              group=self.group)              # [Nk_recv, H]
        # - received_expert_ids  = all_to_all_variable(sorted_expert_ids.contiguous(),
        #                                              input_splits_list,
        #                                              output_splits_list,
        #                                              group=self.group)              # [Nk_recv]

        # =====================================================================
        # Phase 6: local re-argsort by LOCAL expert_id.
        # Received tensor is (source_rank, source-sort-order); each source's block
        # is sorted by global expert_id, but across sources local expert_ids
        # interleave. To get contiguous per-local-expert slices, re-sort locally.
        # =====================================================================
        # TODO(you):
        # - received_local_ids = received_expert_ids - self.expert_start              # [0, experts_per_rank)
        # - sorted_local_ids, local_sort_perm = torch.sort(received_local_ids, stable=True)
        # - local_sorted_x = received_x[local_sort_perm]                              # contiguous per local expert
        # - local_counts   = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
        # - local_offsets  = F.pad(local_counts.cumsum(0), (1, 0))                    # [experts_per_rank + 1]

        # =====================================================================
        # Phase 7: local expert compute — loop over MY experts only.
        # =====================================================================
        # TODO(you):
        # - local_expert_out = torch.empty_like(local_sorted_x)
        # - for local_e in range(self.experts_per_rank):
        #       s = local_offsets[local_e].item()
        #       e = local_offsets[local_e + 1].item()
        #       if s == e:
        #           continue                    # this local expert got no tokens
        #       local_expert_out[s:e] = self.experts[local_e](local_sorted_x[s:e])

        # =====================================================================
        # Phase 8: reverse the local sort — put outputs back in received order
        # (so combine's routing back to originators is a straight inverse of dispatch).
        # =====================================================================
        # TODO(you):
        # - unsorted_received_out = torch.empty_like(local_expert_out)
        # - unsorted_received_out[local_sort_perm] = local_expert_out
        #   Safe scatter: local_sort_perm is a permutation (unique indices).

        # =====================================================================
        # Phase 9: COMBINE — all_to_all_variable (reverse). Swap the splits!
        # =====================================================================
        # TODO(you):
        # - returned_out = all_to_all_variable(unsorted_received_out,
        #                                      output_splits_list,       # ← swapped
        #                                      input_splits_list,        # ← swapped
        #                                      group=self.group)          # [local_N * k, H]

        # =====================================================================
        # Phase 10: weight multiply + local scatter — on the ORIGINATOR rank.
        # =====================================================================
        # TODO(you):
        # - returned_out = returned_out * sorted_weights[:, None]                     # one big multiply
        # - local_y_flat = torch.zeros(local_N, self.hidden,
        #                              device=x.device, dtype=x.dtype)
        # - local_y_flat.index_add_(0, sorted_local_token_ids, returned_out)          # one big scatter

        # =====================================================================
        # Phase 11: all_gather to reassemble full output on every rank.
        # =====================================================================
        # TODO(you):
        # - y_flat = torch.empty(N, self.hidden, device=x.device, dtype=x.dtype)
        # - dist.all_gather_into_tensor(y_flat, local_y_flat, group=self.group)
        # - return y_flat.reshape(original_shape)

        raise NotImplementedError
