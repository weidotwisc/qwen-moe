# Exercise 7 — TP-4 × DP-2 × EP-8 hybrid block

**Goal**: implement one full transformer block under a topology where
**three parallelism axes cross-cut on the same rank set**. Compose
attention TP-4 (from Ex04) with MoE EP-8 (new here), wired with
RMSNorm + residuals to match a Qwen3-style pre-norm layer.

This is the paper's centerpiece: **the case (3) topology** where
`ex06_ep`'s dispatch/striping/gather pattern earns its keep — with
the collective scopes labeled correctly this time (TP-scoped for
striping and gather; EP-scoped for dispatch and combine).

## Topology

On 8 GPUs:

```
DP-2 (outer, no cross-replica comm during forward):
    replica_a = {0, 1, 2, 3}          replica_b = {4, 5, 6, 7}
    processes batch[0:N/2]            processes batch[N/2:N]

TP-4 (per-replica, attention sub-groups):
    tp_group_a = {0, 1, 2, 3}         tp_group_b = {4, 5, 6, 7}
    attention TP-4 within             attention TP-4 within

EP-8 (shared across all 8 ranks, MoE experts):
    ep_group = {0, 1, 2, 3, 4, 5, 6, 7}
    128/8 = 16 experts per rank; dispatch crosses TP-group boundaries
```

**Rank 0 sits in `tp_group_a` AND `ep_group`.** Its collective sequence
per block:

```
attn AR                (tp_group_a)
→ splits negotiation   (ep_group)
→ dispatch × 2         (ep_group)
→ combine              (ep_group)
→ all_gather           (tp_group_a)
```

The paper's composition theorem proves: **this sub-group interleaving
cannot deadlock under fixed-schedule + explicit-group + progress-
preserving-loop discipline**. Every rank issues the same sequence
with its own tp_group handle; no rank stalls waiting for a peer in a
different collective.

## The three cases, and where Ex07 sits

Given the invariant `TP × DP = EP = world_size`:

| Case | Config | Attention output | MoE strategy | Exercise |
|---|---|---|---|---|
| (1) | TP=8, DP=1 | Replicated across all 8 | Lean (filter + all_reduce) | `ex06_ep/reference_lean.py` |
| (2) | TP=1, DP=8 | Distinct per rank | Pure dispatch (no striping, no gather) | `ex06_ep_pure/` |
| **(3)** | **TP=4, DP=2** | **Replicated within TP-4; distinct across DP** | **Stripe-in-TP → dispatch-in-EP → gather-in-TP** | **`ex07_tp_ep_hybrid/` (this)** |

Cases (1) and (2) are degenerate limits of case (3):
- Case (1): `tp_group == ep_group == world`. Striping and gather collapse to identity; dispatch becomes redundant. Lean specialization wins.
- Case (2): `tp_size == 1`. Striping is trivial (each TP group has 1 rank); all_gather is identity. Reduces to `ex06_ep_pure`.

**Only case (3) exercises the full three-axis composition.** That's what Ex07 verifies.

## The 12-phase pipeline

`HybridBlock.forward(x)`:

```
─── Phase A0: input contract check ───────────────────────  [pre-condition]
     x on rank r = [B, T, H] where B = BATCH/DP_SIZE = 1.
     Replicated across ranks in this TP group.

─── Phase A1: RMSNorm on x ───────────────────────────────  [local, elementwise]
     Same weight across tp_group (replicated); output stays replicated.

─── Phase A2: attention TP-4 within tp_group ─────────────  [collective, tp_group]
     TPGQA(hidden, ...): QKV col-parallel → SDPA per head → O row-parallel + AR
     Post-condition: [B, T, H] replicated within tp_group.

─── Phase A3: residual add ────────────────────────────────  [local]
     h = x + attn(rmsnorm(x))

─── Phase B1: RMSNorm on h ───────────────────────────────  [local]
─── Phase B2: HybridMoE.forward(rmsnorm(h)) ──────────────  [multi-collective]
     Phases 0-11 of MoE (below).
─── Phase B3: residual add ────────────────────────────────  [local]
     y = h + moe(rmsnorm(h))

─── Return y ─────────────────────────────────────────────  [B, T, H] replicated within tp_group
```

`HybridMoE.forward(x)`:

```
─── Phase 0: STRIPE within TP GROUP ──────────────────────  [local]
     local_x = x[tp_rank * local_N : (tp_rank + 1) * local_N]

─── Phase 1: local router on local_x ─────────────────────  [local]
─── Phase 2: local argsort by GLOBAL expert_id ──────────  [local]

─── Phase 3: per-EP-destination-rank splits ─────────────  [local]
     dest_ranks = sorted_expert_ids // experts_per_rank
     input_split_sizes = bincount(dest_ranks, minlength=ep_size)

─── Phase 4: NEGOTIATE splits ────────────────────────────  [collective, EP GROUP]
     dist.all_to_all_single(output_splits, input_splits, group=ep_group)

─── Phase 5: DISPATCH × 2 ────────────────────────────────  [collective, EP GROUP]
     received_x           = all_to_all_variable(sorted_x, ...)
     received_expert_ids  = all_to_all_variable(sorted_expert_ids, ...)

─── Phase 6: local re-argsort by local expert_id ────────  [local]
─── Phase 7: local expert compute (loop) ────────────────  [local]
─── Phase 8: reverse local sort ─────────────────────────  [local]

─── Phase 9: COMBINE ─────────────────────────────────────  [collective, EP GROUP]
     returned_out = all_to_all_variable(unsorted, output_splits, input_splits)

─── Phase 10: weight multiply + local scatter ────────────  [local]
     local_y_flat = zeros(local_N, H)
     local_y_flat.index_add_(0, sorted_token_ids, weighted)

─── Phase 11: ALL_GATHER ─────────────────────────────────  [collective, TP GROUP]
     y_flat = empty(N_tp, H)
     dist.all_gather_into_tensor(y_flat, local_y_flat, group=tp_group)

─── Return y ─────────────────────────────────────────────  [N_tp, H] replicated within TP GROUP
```

**Collectives per block: 6** (attn AR + splits + dispatch × 2 + combine + gather).

**The scope split**: dispatch/combine + splits negotiation → EP group;
striping and final all_gather → TP group. This is the exact pattern
`ex06_ep/reference.py` implemented, but with collectives labeled at
the correct scope (`tp_group` for stripe/gather, `ep_group` for
dispatch/combine). Under case (1) with `tp_group == ep_group == world`,
this pattern degenerates to `ex06_ep`'s original bad design (over-scoped
gather). Under case (3), the scopes differ — the pattern is correct
and load-bearing.

## Verification claims — four property classes

**(1) Deadlock freedom under sub-group interleaving.** Every rank issues
the same collective sequence with its own tp_group handle:

```
  tp_group.all_reduce (from o_proj row-parallel)
→ ep_group.all_to_all_single (splits)
→ ep_group.all_to_all_variable (dispatch x)
→ ep_group.all_to_all_variable (dispatch expert_ids)
→ ep_group.all_to_all_variable (combine)
→ tp_group.all_gather_into_tensor (final)
```

**Load-bearing property**: at any hop through this schedule, all peers
in the addressed group hit the same call at the same program point.
No rank is blocked in a tp collective while another rank is trying
to reach an ep collective ahead of it.

The paper's Verus/Dafny proof discharges this via a **causal-order
lemma**: the block-level schedule is a total order (no branching), and
each collective's peer set is either tp_group or ep_group (fixed at
init time). Interleavings between the two groups reduce to standard
sequential collective composition per rank.

**(2) Data-race freedom.** `index_add_` on CUDA uses atomics —
non-deterministic reduction order, abstracted in the spec as a
commutative-associative sum event. Every collective's implementation
inside NCCL is race-free by construction.

**(3) Functional correctness.** HybridBlock's output on rank r
matches `RefBlock(x_full)[dp_rank * local_B : (dp_rank+1) * local_B]`
up to fp reduction-order noise. Tolerance:
- fp32: ~1e-5 baseline × 5 (attention through GQA + MoE combine sum)
- bf16: ~5e-2 baseline × 8 (same, plus bf16 rounding)

**(4) Load-balance / progress.** Empty local experts on any rank
`continue` in the compute loop while remaining collectives still fire
at fixed positions in the schedule. Progress guaranteed for all peers
regardless of routing distribution. Adversarial routing skew produces
bandwidth imbalance (rank 0 receives most dispatched tokens) but no
deadlock — the fixed-schedule invariant holds.

## Configuration used in tests

Match Qwen3-30B-A3B's structure at scaled-down dims for fast tests:

- `HIDDEN=128, INTERMEDIATE=64`
- `N_HEADS=8, N_KV_HEADS=4, HEAD_DIM=32` (mirrors 32/4 GQA ratio)
- `NUM_EXPERTS=8, TOP_K=2` (mirrors 128/8 sparsity)
- `BATCH=2, SEQ=16, N_TOKENS=32` global
- `TP_SIZE=4, DP_SIZE=2, EP_SIZE=8`
- `dtype ∈ {fp32, bf16}`

Each TP group receives `BATCH/DP_SIZE = 1` batch element (16 tokens per group).

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex07_tp_ep_hybrid.py -v
```

Requires all 8 GPUs. Test parametrized on `dtype ∈ {fp32, bf16}`;
reference passes both under `USE_REFERENCE=1`.

## Comparison to nanovllm-jun's topology

Nanovllm-jun uses `tp_size == ep_size == world_size` (case 1). Under
that topology, the entire dispatch cycle is redundant (see
`ex06_ep/reference_lean.py`). **Ex07 targets a richer topology** to
demonstrate the composition theorem where sub-groups genuinely
cross-cut. The paper's argument:

1. **Under case (1)** (nanovllm-jun's default): the lean variant is
   preferable. Our benchmarks show 1.6–4.1× speedup at multi-node
   scale.
2. **Under case (3)** (Ex07's target): dispatch/striping/gather is
   necessary. The composition theorem certifies its correctness.
3. **The paper's contribution**: identifying that case (1) is
   currently deployed with case (3)'s schedule (an inherited
   inefficiency), and providing the formal composition theorem
   that generalizes to all three cases.

## What Ex08 will add

Ex08 (fused Triton kernel) replaces the Python for-loop in Phase 7
of `HybridMoE` with a single grouped-GEMM kernel launch. Same
collective schedule; different local compute. The kernel operates on
the `[N_recv, H]` buffer + `local_offsets` — exactly the layout Phase
6 produces.

## Files

- [solution.py](solution.py) — `HybridBlock` + `HybridMoE` stubs with detailed TODOs.
- [reference.py](reference.py) — working implementation.
- [../ref/block.py](../ref/block.py) — single-GPU `RefBlock` (RMSNorm + RefGQA + RefSparseMoE + residuals).
- [../tests/test_ex07_tp_ep_hybrid.py](../tests/test_ex07_tp_ep_hybrid.py) — 8-GPU spawn test.
