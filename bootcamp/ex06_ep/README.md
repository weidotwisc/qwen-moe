# Exercise 6 — Expert Parallelism (EP) with all-to-all dispatch/combine

**Goal**: distribute the MoE compute across `ep_size` GPUs by
partitioning the E experts across ranks. Each rank owns `E / ep_size`
experts and participates in two `all_to_all_variable` collectives
per forward — one to dispatch tokens to their expert-owning ranks,
one to combine the outputs back.

**Position in the bootcamp**: Ex05b introduced the permuted layout
(`sorted_x`, `sorted_token_ids`, `sorted_weights`, offsets). Ex06
takes that same data path and inserts **two collectives** around the
local expert loop. **The compute part of the algorithm is unchanged.**

This is the first exercise where a single MoE forward requires two
NCCL rounds. Everything about the permutation vocabulary from Ex05b
carries over — the only new element is the pair of collectives.

## Recap — MoE math (from Ex05)

$$
Y_i = \sum_{j=1}^{k} w_{ij}\, f_{e_{ij}}(X_i)
$$

Same routing + weighted combine. EP doesn't change what's computed;
it changes **who computes each term** and **how the sum is assembled**.

## Configuration

Test-scale config (small to run fast):

| Param | Test value | Qwen3-30B-A3B |
|---|---|---|
| `hidden` (H) | 128 | 2048 |
| `intermediate` (I) | 64 | 768 |
| `num_experts` (E) | 8 | 128 |
| `top_k` (k) | 2 | 8 |
| `ep_size` | ∈ {1, 2, 4, 8} | 8 |
| `experts_per_rank` | E / ep_size ∈ {8, 4, 2, 1} | 16 |
| `N` (tokens) | 32 | ~1024+ |

**Divisibility requirement**: `num_experts % ep_size == 0` and
`N % ep_size == 0`. The first partitions experts cleanly; the second
partitions tokens cleanly.

## The data-flow model — pure EP with contiguous token partition

**Input**: `x: [N, H]`, **replicated** across all `ep_size` ranks in
the EP group. Every rank has the same input tensor.

**Per-rank ownership**:
- Rank $r$ owns **experts** `[r * experts_per_rank, (r+1) * experts_per_rank)`.
- Rank $r$ owns **tokens** `[r * (N / ep_size), (r+1) * (N / ep_size))`
  for routing decisions and final output. (Contiguous partition; a
  strided variant appears in Ex07.)

The token partition is what avoids redundant compute — without it,
every rank would dispatch the same records to the same destinations,
wasting `ep_size ×` bandwidth. In Ex07, the TP+EP hybrid uses
strided partitioning within the TP group instead.

**Output**: `y: [N, H]`, **replicated** across all ranks. Reconstructed
via an `all_gather` at the end of forward.

## The 11-phase pipeline

Each phase's ownership is labeled:
**\[replicated]** = same on every rank; **\[local]** = per-rank distinct;
**\[collective]** = crosses rank boundary.

```
─── Phase 0: local token partition ───────────────────────  [local]
     local_x = x[ep_rank * local_N : (ep_rank+1) * local_N]        # [local_N, H]

─── Phase 1: local router (same as Ex05b, on local_x) ────  [local]
     router_logits    = self.gate(local_x)                         # [local_N, E]
     top_k_weights,
     top_k_experts    = softmax + topk + optional renorm           # [local_N, k]

─── Phase 2: local argsort by GLOBAL expert_id ───────────  [local]
     sorted_expert_ids, sort_perm = torch.sort(top_k_experts.flatten())
     sorted_local_token_ids       = local_token_ids[sort_perm]     # [local_N * k]
     sorted_weights               = weights_flat[sort_perm]        # [local_N * k]
     local_x_permuted             = local_x[sorted_local_token_ids]  # [local_N * k, H]

─── Phase 3: compute per-destination-rank splits ─────────  [local]
     dest_ranks         = sorted_expert_ids // experts_per_rank
     input_split_sizes  = bincount(dest_ranks, minlength=ep_size)  # [ep_size]

─── Phase 4: negotiate output splits ─────────────────────  [collective]
     all_gather_into_tensor(input_split_sizes, ep_group) → [ep_size × ep_size]
     output_split_sizes = column ep_rank of the gathered matrix    # [ep_size]

─── Phase 5: DISPATCH — all_to_all_variable × 2 ──────────  [collective]
     received_x           = all_to_all(local_x_permuted, in_splits, out_splits)
     received_expert_ids  = all_to_all(sorted_expert_ids, in_splits, out_splits)
     # sorted_weights STAYS local — never leaves this rank.

─── Phase 6: local re-argsort by LOCAL expert_id ─────────  [local]
     received_local_ids           = received_expert_ids - expert_start  # [0, experts_per_rank)
     sorted_local_ids, local_sort = torch.sort(received_local_ids)
     local_sorted_x               = received_x[local_sort]         # contiguous per local expert
     local_counts   = bincount(sorted_local_ids, minlength=experts_per_rank)
     local_offsets  = F.pad(local_counts.cumsum(0), (1, 0))        # [experts_per_rank + 1]

─── Phase 7: local expert compute loop ──────────────────  [local]
     local_expert_out = empty_like(local_sorted_x)
     for local_e in range(experts_per_rank):
         s, e = local_offsets[local_e], local_offsets[local_e + 1]
         if s == e: continue
         local_expert_out[s:e] = self.experts[local_e](local_sorted_x[s:e])

─── Phase 8: reverse the local sort ─────────────────────  [local]
     unsorted_received_out               = empty_like(local_expert_out)
     unsorted_received_out[local_sort]   = local_expert_out        # scatter back

─── Phase 9: COMBINE — all_to_all_variable (reverse) ────  [collective]
     returned_out = all_to_all(unsorted_received_out,
                                out_splits, in_splits)              # swap splits!

─── Phase 10: weight multiply + local scatter ───────────  [local]
     returned_out    *= sorted_weights[:, None]                    # ONE big multiply
     local_y_flat     = zeros(local_N, H)
     local_y_flat.index_add_(0, sorted_local_token_ids, returned_out)  # ONE big scatter

─── Phase 11: all_gather to reassemble full output ──────  [collective]
     y_flat = empty(N, H)
     all_gather_into_tensor(y_flat, local_y_flat, ep_group)
```

**Four collectives per forward** (one `all_gather_into_tensor` for
splits, two `all_to_all_variable` for dispatch, one for combine, one
`all_gather_into_tensor` for the final assembly). All are fixed-schedule
— every rank issues the same sequence regardless of routing data,
which is the deadlock-freedom property we're preserving from Ex00.

## Why weights stay on the originator

The design here uses **Choice A** from the crosscutting discussion —
weights and `sorted_token_ids` never cross the network:

- Dispatch payload: `received_x` (hidden states) + `received_expert_ids`
  (routing metadata). **Not** weights.
- Combine payload: raw expert output. **Not** weight-multiplied.
- Multiply-by-weights + scatter happen on the ORIGINATING rank in Phase 10.

Two reasons:

1. **Slimmer collective payloads**: hidden states + expert_ids only.
   Weights (which are per-record scalars) would add ~1/H fractional
   bandwidth, but the principle of "compute is weight-less" is cleaner.
2. **Cleaner Verus/Dafny spec**: each expert is a **pure function**
   $\mathbb{R}^H \to \mathbb{R}^H$. Routing weights are a property of
   the token-router interaction, not the token-expert interaction. Keeping
   them out of the compute rank preserves the pure-function view.

An alternative (**Choice B**, what nanovllm-jun uses) sends weights
with the dispatch, applies them on the compute side inside the expert
loop, and returns weighted outputs. Also correct, slightly more perf
under some kernels, but semantically muddier.

## Traps to watch for

- **`input_split_sizes` on rank i must match `output_split_sizes[i]`
  on rank j** where rank i sends to rank j. This is
  `all_to_all_variable`'s load-bearing precondition. If it's violated
  by even one (i, j) pair, NCCL will deadlock or silently corrupt.
  Phase 4's `all_gather_into_tensor` negotiation is the standard way
  to satisfy this — never construct splits ad-hoc without gathering.
- **Split tensors need `.tolist()`** before passing to
  `all_to_all_single` / `all_to_all_variable`. The API wants Python
  lists (or `list[int]`), not GPU tensors. Materializing the tensor
  to a list triggers a GPU-CPU sync — unavoidable, and the reason
  dispatch has a small blocking point.
- **`torch.sort(..., stable=True)`** on the local re-argsort (Phase 6)
  gives reproducibility. `bincount` doesn't care about order, but the
  reverse-sort in Phase 8 needs a consistent permutation.
- **Empty local experts**: if some ranks receive zero tokens for a
  particular local expert, the loop's `if s == e: continue` handles
  it. Under EP-8 with test-scale N=32, this happens routinely (the
  test is designed to hit it).
- **`unsorted_received_out[local_sort_perm] = local_expert_out`** —
  this is a scatter WRITE, but `local_sort_perm` is a permutation
  (unique indices), so no race. Safe on CUDA.
- **`torch.bincount` overload for int32 vs int64**: pass a `torch.long`
  tensor to be safe. Split sizes should always be `torch.long`.
- **`.contiguous()` on tensors before all_to_all**: the collective
  reads memory in stride order. If your tensor has non-standard
  strides (from a view), NCCL may error or corrupt.
- **Explicit `group=self.group`** on every collective — Ex07's hybrid
  makes this critical. Get the habit now.

## Numerical tolerance

The EP output must match `RefSparseMoE`'s single-GPU output up to
**fp reduction-order noise**. Sources of divergence:

1. **`index_add_` accumulation order** — CUDA atomics are non-deterministic
   for the same reason FA2 backward is: multiple threads adding into
   the same output slot, order not guaranteed.
2. **`all_to_all_variable` reduction order** — no reduction happens in
   dispatch/combine (just data movement), so this isn't a source.
3. **Different iteration order over experts** — reference iterates
   0..E-1 in a single loop; EP iterates 0..experts_per_rank on each
   rank in parallel. Order-of-adds differs.

Expect `atol=1e-4, rtol=1e-4` in fp32, `atol≈5e-2` in bf16 (matching
the Ex05b tolerance).

## Config used in tests

- `HIDDEN = 128`, `INTERMEDIATE = 64`
- `NUM_EXPERTS = 8`, `TOP_K = 2`
- `N_TOKENS = 32`, `BATCH = 2`, `SEQ = 16`
- `ep_size ∈ {1, 2, 4, 8}`, `dtype ∈ {fp32, bf16}`

At Qwen3-30B-A3B scale, `num_experts = 128`, `top_k = 8`, `ep_size = 8`.
The algorithm is identical — just larger constants.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex06_ep.py -v
```

Requires `ep_size` GPUs available. Uses `run_on_ranks` from `dist_utils.py`
to spawn processes.

## What Ex07 will add

Ex07 turns `EPSparseMoE` into a **HybridBlock** by:

1. **Splitting the world into two NCCL groups**:
   - `tp_group_a` = ranks {0, 1, 2, 3} — TP-4 for attention on the first half.
   - `tp_group_b` = ranks {4, 5, 6, 7} — TP-4 for attention on the second half.
   - `ep_group`   = all 8 ranks — EP-8 for MoE.
2. **Attention runs under TP** inside each TP group (Ex04's TPGQA).
3. **MoE runs under EP** across the whole world (this Ex06 code).
4. **Striped-token trick within a TP group**: each rank owns tokens
   `[r::tp_size]` (strided) so the MoE compute isn't duplicated across
   the TP-replicated hidden states.
5. **`all_gather` within the TP group** at the end of MoE to
   reconstruct the full token set.

The Ex06 code we're writing is directly reused as the MoE block inside
Ex07's HybridBlock. **Ex07 is Ex04 + Ex06 wired together, plus the
striped-token trick.**

## What Ex08 will add

Ex08 replaces the Python for-loop in Phase 7 with a single Triton
grouped-GEMM kernel. Same `local_sorted_x`, `local_offsets`, same
`local_expert_out` shape — the kernel just internalizes the loop.
No atomics needed in the kernel body (per-expert blocks write to
disjoint slices of `local_expert_out`), matching FA2 forward's
kernel structure.

## Files

- [solution.py](solution.py) — `EPSparseMoE` stub with detailed TODOs.
- [reference.py](reference.py) — working implementation.
- [test file](../tests/test_ex06_ep.py) — 8-GPU spawn test comparing to `RefSparseMoE`.
