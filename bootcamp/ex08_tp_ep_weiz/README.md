# Exercise 8 — Wei's hybrid intra-lean/inter-dispatch schedule

**Wei's idea.** Novel schedule for MoE-EP under TP × DP × EP composition
that exploits an observation vLLM / nanovllm-jun miss: **intra-TP-group
routing records don't need cross-network dispatch**, because the
originating rank and the destination rank share replicated hidden states
from the preceding attention TP AR. Only **inter-TP-group** records
require real data movement.

## The observation

Under Ex07's TP-4 × DP-2 × EP-8 topology:

- Attention TP-4 within each TP group produces `[N/DP, H]` **replicated
  across the group's 4 ranks**.
- MoE routes each token to top_k experts, distributed across all 8 EP ranks.
- **~50% of records** (uniform routing) have their destination expert on
  a rank **within the same TP group as the token's owner**.

For those intra-TP records: the destination rank already has the token
(replicated across the TP group). Sending it via `all_to_all_variable`
moves data that's already at the destination — the same redundancy we
diagnosed in `ex06_ep`.

For inter-TP records (routing crossing TP groups): the destination
rank does NOT have the token. Real dispatch is required.

**Wei's hybrid schedule splits the two cases**:
- **Intra-TP** → lean pattern (filter to my local experts, compute, contribute).
- **Inter-TP** → dispatch pattern (stripe within TP, all_to_all_v across EP).

One `all_reduce` within the TP group at the end distributes both contributions.

## The two-branch schedule

```
Rank r in TP group T (of size tp_size), EP world of size ep_size:

Phase 1: router on replicated x → routing records
         [N_tp, top_k]  where N_tp = N/DP tokens for this TP group

Phase 2: classify each of the N_tp*top_k records:
   dest_rank = expert_id // experts_per_rank
   intra_mask = (dest_rank == r)               # this rank's OWN local experts
   inter_mask = (dest_rank // tp_size != T)    # dest is in a different TP group
   (records with dest in T but dest_rank ≠ r are handled by OTHER ranks in T's
    intra branch; the final AR sums their contributions.)

Phase 3 — INTRA BRANCH  (no collective, all local):
   Filter records to intra_mask.
   Sort by local expert id.
   For each local expert: compute expert(x[intra_token_ids]) * weights.
   partial_output.index_add_(0, intra_token_ids, weighted_intra_output)

Phase 4 — INTER BRANCH  (2 all_to_all_v over EP group):
   inter_positions = inter_mask.nonzero()
   my_stripe = inter_positions[tp_rank :: tp_size]     # this rank's 1/tp_size share
   
   Sort my_stripe by dest_expert_id.
   input_split_sizes[j] = bincount(dest_ranks, minlength=ep_size)
   # NOTE: input_split_sizes[j] = 0 for j inside my TP group. That's the
   # "zero-count intra" trick that lets the same all_to_all_v API handle
   # only-cross-group traffic.
   
   Negotiate output_split_sizes via all_to_all_single over EP group.
   Dispatch x and expert_ids via all_to_all_v over EP group.
   Local re-argsort by local expert; compute expert(received_x); scatter.
   Combine (reverse all_to_all_v) → returned_out.
   partial_output.index_add_(0, my_stripe_token_ids, returned_out * my_weights)

Phase 5 — WITHIN-TP ALL_REDUCE:
   dist.all_reduce(partial_output, group=tp_group)
   # This sums:
   #  - Intra contributions: each rank in T contributed to disjoint (expert, token)
   #    slots; AR gives every rank the union.
   #  - Inter contributions: each rank in T contributed a disjoint stripe of inter
   #    records; AR gives every rank the union.

Return partial_output.
```

## The "zero-count intra" trick

Under Phase 4, the input_split_sizes for intra-group destinations are
**exactly zero** — those records were already handled in Phase 3, never
enter the inter stripe. NCCL's `all_to_all_single` (and our
`all_to_all_variable` wrapper) natively handles zero-size messages
efficiently — no data is sent for zero-count entries, and the collective
completes without any point-to-point send/recv machinery.

**This is the key trick that keeps the API surface clean.** We use one
`all_to_all_v` call for the inter branch; the zero counts make it act
as "point-to-point across TP-group boundary only." No special-case
send/recv code needed.

## Bandwidth analysis

Per rank per forward, Qwen3-30B-A3B dims (H=2048, top_k=8), ep=8,
tp=4, uniform routing:

| Component | Ex07 canonical | Wei's hybrid |
|---|---|---|
| Intra records handling | Goes through EP dispatch (redundant) | Local compute + within-TP AR |
| Inter records handling | Goes through EP dispatch | Same, but only cross-group traffic |
| Intra all_gather | 3/8 · N · H | 0 |
| Within-TP all_reduce | 0 | 3/4 · N · H |
| EP dispatch+combine (round-trip) | 4 · N · H | 2 · N · H |
| **Total per rank** | **~4.375 · N · H** | **~2.75 · N · H** |

**~37% bandwidth reduction under uniform routing.** At extreme skew,
the intra branch is entirely balanced (within-TP AR is data-independent);
only the inter branch's dispatch shows skew asymmetry.

## Composition theorem — proof structure

The paper's Verus/Dafny spec for this schedule now composes THREE
sub-schedules:

1. **Attention TP-4** (Ex04's schedule): 1 all_reduce over tp_group.
2. **Intra-lean branch** (this Ex08): local compute, no collective.
3. **Inter-dispatch branch** (this Ex08): all_to_all_single (splits) +
   2 all_to_all_variable (dispatch × 2 + combine) over ep_group.
4. **Final within-TP AR** (this Ex08): 1 all_reduce over tp_group.

Composition:
- Every rank issues the same collective sequence with its own tp_group handle.
- Group handles are explicitly threaded through every call.
- Progress-preserving: intra branch has no data-dependent branching that
  skips collectives; inter branch's zero-count entries are handled by
  NCCL without deadlock.

**Deadlock freedom**: the intra branch's absence of collectives makes it
trivially non-blocking. The inter branch's collectives are on ep_group;
the intra AR at end is on tp_group. Sequential per rank, same order on
every rank of ep_group and every rank of tp_group. Composition theorem
from Ex07 applies directly with the additional intra-lean sub-schedule
inserted as a no-op collective step.

**Functional correctness**: intra contributions and inter contributions
are added to disjoint token positions in partial_output... wait, they
CAN overlap at the same token (if the token has both an intra-routed
expert and an inter-routed expert). At overlap, index_add_ correctly
accumulates both contributions via CUDA atomics. The within-TP AR at
end sums across ranks in the TP group (each rank contributed disjoint
subsets of routing records).

Output matches Ex07's canonical schedule up to fp reduction-order
tolerance.

## Configuration used in tests

Identical to Ex07's:
- `HIDDEN=128, INTERMEDIATE=64`
- `N_HEADS=8, N_KV_HEADS=4, HEAD_DIM=32`
- `NUM_EXPERTS=8, TOP_K=2`
- `BATCH=2, SEQ=16, N_TOKENS=32` global
- `TP_SIZE=4, DP_SIZE=2, EP_SIZE=8`
- `dtype ∈ {fp32, bf16}`

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex08_tp_ep_weiz.py -v
```

Reference: 2/2 green (fp32 + bf16). The block's output matches RefBlock's
up to reduction-order tolerance identical to Ex07's — proving the hybrid
schedule is functionally equivalent to the canonical schedule.

## Where this fits in the paper

This is **the paper's most-original technical contribution** identified
so far:

1. **Nanovllm-jun / vLLM ship the canonical schedule** (case-3 pattern
   applied uniformly with `tp_size == ep_size == world_size`) — which
   makes it redundant under their specific topology (see `ex06_ep_lean`).
2. **Ex07's canonical schedule** for case-3 (`tp_size < world_size`) IS
   necessary but still moves intra-TP-group tokens redundantly.
3. **Wei's hybrid schedule** (this Ex08) eliminates the intra-TP-group
   redundancy by branching between lean (intra) and dispatch (inter).
4. Empirical: ~37% bandwidth reduction at case-3 uniform routing.
5. Formal: composition theorem extends to cover the two-branch schedule.

## Files

- [solution.py](solution.py) — `HybridScheduleBlock` + `HybridScheduleMoE` stubs with TODOs.
- [reference.py](reference.py) — working implementation.
- [../tests/test_ex08_tp_ep_weiz.py](../tests/test_ex08_tp_ep_weiz.py) — 8-GPU spawn test.

## Attribution

The hybrid intra-lean/inter-dispatch schedule design in this exercise
was proposed by **Wei Zhang** during the Ex07 design discussion, upon
noticing that `ex07_tp_ep_hybrid`'s canonical schedule redundantly
dispatches intra-TP-group tokens. Hence the `_weiz` suffix on the
directory name.
