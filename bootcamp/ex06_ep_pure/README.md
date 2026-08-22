# Exercise 6 (pure EP) — DP-partitioned inputs, dispatch load-bearing

**Goal**: implement Expert Parallelism in its natural setup — each
rank owns a **distinct** 1/ep_size share of the batch tokens (as if
a DP layer preceded the MoE). Expert compute is sharded across all
EP ranks. Each rank keeps its own share of the final output.

Under this setup, dispatch is **structurally necessary**: each rank
holds a disjoint subset of tokens, so a token owned by (say) rank 0
cannot be processed by an expert on rank 5 unless rank 0 sends it
via `all_to_all_variable`. No `all_reduce` shortcut exists because
non-owning ranks have no data to contribute for tokens they don't
hold.

## Relationship to `ex06_ep/`

[../ex06_ep/](../ex06_ep/) uses a **replicated input** pre-condition:
every rank has the full `[N, H]` tensor. Under that pre-condition,
each rank could filter to its LOCAL experts and skip dispatch
entirely (see [../ex06_ep/reference_lean.py](../ex06_ep/reference_lean.py)) —
dispatch becomes a redundant collective moving data that's already at
the destination.

**That replication is a TP-scope property, not an EP-scope property.**
It arises only when `tp_size == ep_size == world_size` (nanovllm-jun /
vLLM default). Under production topologies with genuine EP (DP × EP,
or PP × EP, or standalone EP-over-distinct-batches), each rank has
distinct data, and dispatch is load-bearing.

**`ex06_ep/` teaches "how nanovllm-jun does EP inside a TP-replicated
group"** — preserving the design bug that dispatches redundantly.

**`ex06_ep_pure/` teaches EP as an algorithm** — with the natural
"distinct data per rank" pre-condition. Dispatch here has real work
to do.

## The scope conflation table

| Setup | Input distribution across ep_group | Dispatch necessary? | Lean shortcut? |
|---|---|---|---|
| **Pure EP (this exercise)** | Distinct per rank (DP-style) | **Yes** | No |
| TP == EP == world (nanovllm-jun) | Replicated (from attn AR) | **No — redundant** | Yes (all_reduce) |
| DP × TP × EP hybrid | Replicated within TP subgroups | Partially | Only within TP |

## The 9-phase pipeline

**Data flow**: each rank owns `local_x: [N/ep_size, H]`, distinct
content. Rank r's tokens are conceptually tokens `[r · local_N, (r+1) · local_N)`
of some global batch — but rank r never sees the other ranks' tokens.

```
─── Phase 0: local input ────────────────────────────────  [local]
     rank r has local_x: [local_N, H]  (distinct across ranks)

─── Phase 1: local router ───────────────────────────────  [local]
     router_logits    = self.gate(local_x)                       # [local_N, E]
     top_k_weights,
     top_k_experts    = softmax + topk (+ optional renorm)       # [local_N, k]

─── Phase 2: local argsort by GLOBAL expert_id ──────────  [local]
     sorted_expert_ids, sort_perm = torch.sort(top_k_experts.flatten())
     sorted_weights   = weights_flat[sort_perm]                  # [local_N * k]
     sorted_local_token_ids = arange(local_N).repeat_interleave(k)[sort_perm]
     sorted_x         = local_x[sorted_local_token_ids]          # [local_N * k, H]

─── Phase 3: compute per-destination-rank splits ────────  [local]
     dest_ranks          = sorted_expert_ids // experts_per_rank
     input_split_sizes   = bincount(dest_ranks, minlength=ep_size)   # [ep_size]

─── Phase 4: negotiate output splits ────────────────────  [collective]
     all_to_all_single(output_splits, input_splits)                  # [ep_size]

─── Phase 5: DISPATCH — all_to_all_variable × 2 ─────────  [collective]
     received_x           = a2a_v(sorted_x, ...)                     # [Nk_recv, H]
     received_expert_ids  = a2a_v(sorted_expert_ids, ...)             # [Nk_recv]
     (weights + sorted_local_token_ids stay LOCAL to originator)

─── Phase 6: local re-argsort by local expert_id ────────  [local]
     received_local_ids = received_expert_ids - expert_start
     torch.sort(received_local_ids, stable=True) → local_sort_perm
     local_sorted_x, local_offsets = ...

─── Phase 7: local expert compute ───────────────────────  [local]
     for local_e in range(experts_per_rank):
         local_expert_out[s:e] = self.experts[local_e](local_sorted_x[s:e])

─── Phase 8: reverse local sort ─────────────────────────  [local]
     unsorted_received_out[local_sort_perm] = local_expert_out

─── Phase 9: COMBINE — all_to_all_variable ──────────────  [collective]
     returned_out = a2a_v(unsorted_received_out, output_splits, input_splits)

─── Phase 10: weight multiply + local scatter ───────────  [local]
     weighted = returned_out * sorted_weights[:, None]
     local_y  = zeros(local_N, H)
     local_y.index_add_(0, sorted_local_token_ids, weighted)

─── Phase 11: return local_y (no all_gather!) ───────────  [local]
     Each rank keeps its own [local_N, H] output.
```

**Total collectives: 4** (splits negotiation + 2 dispatch + 1 combine).

**No `all_gather` at the end.** Rank r's output is `local_y: [local_N, H]`
— the results for the tokens rank r originally owned. The
concatenation `[local_y_0, local_y_1, ..., local_y_{ep-1}]` across ranks
would give the full output, but no single rank materializes it.

**No `all_reduce` alternative.** Non-rank-r ranks have no partial
output for token positions owned by rank r — they don't have those
tokens' hidden states. `all_reduce` on an all-zeros buffer wouldn't
recover the data.

## Why this is the natural EP setup

- **Training with DP + EP**: each DP replica has a distinct batch;
  MoE inside a replica sees distinct tokens per rank. Dispatch
  routes to expert-owners; no all_reduce alternative.
- **Multi-node inference with DP over batches**: two nodes each
  serving different request cohorts share MoE experts via EP;
  dispatch moves cohorts to their expert-owners.
- **Pipeline parallel MoE stages**: activations arrive at the MoE
  stage on a per-rank basis (one PP-rank per EP rank).

None of these have replicated input across the EP group. Dispatch is
real work.

## Contrast: ex06_ep vs ex06_ep_pure at the code level

The algorithmic body is **nearly identical**. What differs is only the
outer contract:

|  | `ex06_ep/reference.py` | `ex06_ep_pure/reference.py` (this) |
|---|---|---|
| Input | `x: [N, H]` replicated | `local_x: [N/ep, H]` distinct |
| Slice inside `forward` | Phase 0: `local_x = x[r*local_N:(r+1)*local_N]` | Not needed — already local |
| Final assembly | Phase 11: `all_gather` to `[N, H]` | Not needed — return `local_y` |
| Output | `[N, H]` replicated | `[N/ep, H]` distinct |

Delete Phase 0 (input slicing) and Phase 11 (final all_gather), keep
everything else. **That's the whole difference.** Ex06's replicated-input
version is just this exercise wrapped in a redundant "replicate at
start, gather at end" outer shell.

## Configuration used in tests

- `HIDDEN = 128`, `INTERMEDIATE = 64`
- `NUM_EXPERTS = 8`, `TOP_K = 2`
- `N_TOKENS = 32` (global), so `local_N = N/ep_size` per rank
- `BATCH = 2`, `SEQ = 16` (rank r's slice: `local_N` tokens)
- `ep_size ∈ {1, 2, 4, 8}`, `dtype ∈ {fp32, bf16}`

Each rank generates its own `local_x` from the same global seed
(so we can reconstruct the "full x" in the test oracle and verify
each rank's `local_y` matches the corresponding slice of the reference).

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex06_ep_pure.py -v
```

## Files

- [solution.py](solution.py) — `EPSparseMoE` stub with detailed TODOs.
- [reference.py](reference.py) — working implementation.
- [test file](../tests/test_ex06_ep_pure.py) — 8-GPU spawn test, distinct input per rank.
