# Verus artifact audit guide for Jun

**Purpose**: onboard Jun to the paper's Verus artifact so he can audit
efficiently. Written by Claude (AI drafter). Wei has read this and
signed off on the structure; Jun is the auditor of record.

**Status at time of writing** (2026-09-01):
- **143 verified lemmas, 0 errors** across **15 Verus files**.
- All files typecheck against Verus 0.2025.07.12.0b6f3cb.
- Every proof has been drafted by Claude in this session; Jun's audit
  is the human-verification step of the paper's methodology contribution.

## 1. Audit protocol (read this first)

The paper's methodology claim is: **contracts and proofs are AI-drafted
by Claude and audited by a Verus expert (Jun) — the audit is the
correspondence-check between Verus formalism and Python implementation.**

Concretely, for each file, Jun should answer four questions:

1. **Does it typecheck?** Run `verus <file>.rs` (or
   `verus --crate-type=lib <file>.rs` for the top-level `verus/` files).
   Confirm `verification results:: N verified, 0 errors`.
2. **Are the axioms sound?** Each `#[verifier::external_body]` block is
   a trust point. Read it and confirm the ensures-clause faithfully
   mirrors the corresponding Python code or is a well-known
   mathematical fact.
3. **Are the contracts non-decorative?** For each claimed property,
   ask: "if I delete the semantic clauses and keep only the shape
   clauses, does the proof still go through?" If yes, the semantic
   clauses are decoration and the contract needs strengthening.
4. **Are proofs honest?** No `admit()` calls in the final proof. Every
   `#[verifier::external_body]` maps to either a Python-correspondence
   axiom or a well-known math fact.

## 2. File-by-file directory

### 2.1 Per-component proofs (Tier 1)

Each file lives at `bootcamp/exNN_<name>/verification/<component>.rs`
alongside a `PROPERTIES.md` that states the semantic spec in prose and
math. Read `PROPERTIES.md` first, then the `.rs`.

| Exercise | File | Verified | What it proves | Key axioms trusted |
|----------|------|---------:|----------------|--------------------|
| Ex01 linear_tp (col) | `column_parallel.rs` | 12 | C1-C4 for ColumnParallelLinear | AXIOM_M1 (matmul splits over out-dim) |
| Ex01 linear_tp (row) | `row_parallel.rs` | 13 | R1-R4 for RowParallelLinear + R4b (Replicated after all-reduce) | AXIOM_M2, `axiom_all_reduce_produces_replicated` |
| Ex02 mlp_tp | `mlp_tp.rs` | 13 | M1-M3 merged column + T1-T3 SwiGLU composition | AXIOM_M1, AXIOM_M2, AXIOM_S1 |
| Ex03 mha_tp | `mha_tp.rs` | 13 | Q1-Q4 QKV column + A1-A2 attention composition | AXIOM_M1, AXIOM_ATTN_HEAD_LOCAL |
| Ex04 gqa_tp | `gqa_tp.rs` | 11 | G1-G4 GQA + R1 KV-replication invariants | AXIOM_RI1 (repeat_interleave) |
| Ex05 moe_baseline | `moe_baseline.rs` | 13 | RT1-RT3 routing invariants + E1/E2 stubs | AXIOM_SOFTMAX_SUM |
| Ex06_ep_pure | `ep_pure.rs` | 8 | EP1-EP4 expert-partition + dispatch symmetry | AXIOM_A2A_COUNT_SYMMETRIC |
| Ex06_ep (lean) | `lean.rs` | 7 | L2 disjointness + L2-covers + L3/L5 | AXIOM_ALLREDUCE_SUM_REPLICATED |
| Ex07 tp_ep_hybrid | `hybrid.rs` | 3 | H2 striping determinism (rest stubbed) | tensor_on opaque |
| Ex09 fused_moe (contract) | `fused_moe.rs` | 2 | F1 offset coverage + F3 empty-expert | expert_apply opaque |
| Ex09 fused_moe (DSL) | `fused_kernel_dsl.rs` | 13 | K1a/K1b/K1c tile coverage + K3 derived correctness | AXIOM_MATMUL_SPLITS_OVER_K |
| **Subtotal** | | **108** | | |

**Ex01 detail** — the paper's exemplar per-component proof. Read
[`bootcamp/ex01_linear_tp/verification/PROPERTIES.md`](../bootcamp/ex01_linear_tp/verification/PROPERTIES.md)
first — it argues the three-tool structure (Verus + Dafny + Z3) that
the paper's methodology section names. The Verus file follows this
pattern exactly, and every subsequent per-component .rs mirrors it.

**Ex02 detail** — first composition proof. Ships with an
[`anecdote.txt`](../bootcamp/ex02_mlp_tp/verification/anecdote.txt) —
the audit-cycle log for this file. Verus rejected Claude's initial
draft twice (missing well-formedness scaffolding + missing div_mod
lemma). Read anecdote.txt as evidence that Verus catches real
proof-drafting errors; this is the concrete data point for
Contribution 3.

### 2.2 Shared framework (Tier 2)

`verus/axiom_base.rs` (4 verified lemmas): distributed-state predicates
`Replicated`, `ExpertPartition`, `routing_conserved_local`; the
approx_eq predicate and its three lemmas (refl / sym / trans); trusted
axioms for `broadcast` and `all_reduce`.

**Audit question**: is `approx_eq` defined too weakly? At present it's
shape+dtype-only. Downstream contracts treat it as opaque and never
unfold its definition, so this is fine for the meta-theorem — but the
paper's threats-to-validity section should note that a fully-numerical
`approx_eq` would strengthen the claim.

### 2.3 Composition + safety theorems (Tier 3)

Three files at the repo-level `verus/` directory.

| File | Verified | What it proves |
|------|---------:|----------------|
| `lean_equiv_hybrid_dp1.rs` | 5 | Ex06_ep/lean ≡ Ex07 hybrid, under DP=1 |
| `naive_equiv_fused_moe.rs` | 8 | Ex05 (naive & permuted) ≡ Ex09 fused |
| `composition_theorem.rs` | 18 | Meta-theorem + 2 instances + 2×2 block corollary + 3 safety theorems |
| **Subtotal** | **31** | |

**The meta-theorem** (`composition_theorem.rs` §3,
`theorem_shared_spec_implies_equiv`) is the paper's Contribution 2
headline. Its proof body is 5 lines: symmetry + transitivity of
`approx_eq`. Every specific composition claim in the paper is an
INSTANCE of this meta-theorem — the paper's methodological point.

## 3. Grand total

**130 verified lemmas, 0 errors, 14 files**.

## 4. Property matrix — what's actually proved

### Functional equivalence properties

| Claim | Where | Lemma |
|-------|-------|-------|
| ColumnParallelLinear forward = unsharded matmul | Ex01 col | `c4_forward_correctness_tp2` |
| RowParallelLinear sum-of-partials = unsharded matmul | Ex01 row | `r4_forward_correctness_tp2` |
| RowParallelLinear output is Replicated on tp_group | Ex01 row | `r4b_output_replicated_after_all_reduce` |
| Merged column-parallel = unsharded merged matmul | Ex02 | `m3_merged_forward_correctness_tp2` |
| QKV three-way split matches per-projection matmul | Ex03 | `q3_qkv_forward_correctness_tp2` |
| GQA weight_loader post-condition | Ex04 | `g4_gqa_weight_loader_postcondition` |
| Naive MoE refines MoE_spec | Ex05 | `e1_naive_refines_spec_stub` (external) |
| Permuted MoE refines MoE_spec | Ex05 | `e2_permuted_refines_naive_stub` (external) |
| Fused Triton refines MoE_spec | Ex09 | `f4_fused_matches_python_loop_stub` (external) |
| **META**: A ≈ S ∧ B ≈ S ⟹ A ≈ B | `composition_theorem.rs` §3 | `theorem_shared_spec_implies_equiv` |
| Lean ≡ Hybrid under DP=1 | `lean_equiv_hybrid_dp1.rs` | `theorem_lean_equiv_hybrid_dp1` |
| Permuted ≡ Fused Triton | `naive_equiv_fused_moe.rs` | `theorem_permuted_equiv_fused` |
| Naive ≡ Fused Triton | `naive_equiv_fused_moe.rs` | `theorem_naive_equiv_fused` |
| Naive ≡ Permuted (corollary) | `naive_equiv_fused_moe.rs` | `corollary_naive_equiv_permuted` |
| Block-level 2×2 grid equivalence | `composition_theorem.rs` §6 | `corollary_block_variants_equivalent` |

### Global safety properties (Contribution 2 §7)

| Claim | Where | Lemma | Trusts per-component axioms |
|-------|-------|-------|------------------------------|
| S1 Token conservation, variant-independent | `composition_theorem.rs` §7.1 | `theorem_token_conservation` | Ex05 RT1 + Ex06 EP4 + Ex09 F1 |
| S1 corollary: same record count across variants | `composition_theorem.rs` §7.1 | `corollary_records_variant_invariant` | (as above) |
| S2 Deadlock-freedom, variant-independent | `composition_theorem.rs` §7.2 | `theorem_deadlock_free` | Ex06 EP6 + Ex07 H5 |
| S3 Unique-writer invariant, variant-independent | `composition_theorem.rs` §7.3 | `theorem_unique_writer` | Ex06_ep/lean L3 + Ex05 RT4 + Ex09 F3 |

### Routing / partitioning invariants (per-component)

| Claim | Where | Lemma |
|-------|-------|-------|
| Shard-and-gather roundtrip | Ex01, Ex02, Ex03, Ex04 | `c1_shard_gather_roundtrip`, `m2_w{0,1}_gather_roundtrip`, `q2_{q,k,v}_gather_roundtrip`, `g3_kv_gather_roundtrip` |
| Sharding disjointness | Ex01, Ex06_ep_pure, Ex06_ep/lean | `c2_sharding_disjoint`, `ep1_expert_sharding_disjoint`, `l2_local_mask_disjoint` |
| Sharding coverage | Ex06_ep_pure, Ex06_ep/lean | `ep2_expert_coverage`, `l2_local_mask_covers` |
| Offset monotonicity | Ex05 | `rt2_offset_monotonicity`, `lemma_cumsum_monotone` |
| Token conservation via offsets | Ex05, Ex09 | `rt1_token_conservation`, `f1_offsets_cover_range` |
| Top-k weight normalization | Ex05 | `rt3_topk_weight_normalization` |
| KV replication invariant | Ex04 | `g2_kv_replication_invariant` |
| Weight-loader postcondition (each variant) | Ex01, Ex02, Ex03, Ex04 | `c3_weight_loader_postcondition`, `m1_merged_weight_loader_postcondition`, `q1_qkv_weight_loader_postcondition`, `g4_gqa_weight_loader_postcondition` |
| Dispatch symmetry | Ex06_ep_pure | `ep3_dispatch_symmetric`, `ep4_token_conservation_pairwise` |
| Empty-expert handling | Ex09 | `f3_empty_expert_no_rows` |
| Replication postcondition of all_reduce | Ex06_ep/lean, `verus/axiom_base.rs` | `l5_output_replicated`, `axiom_all_reduce_sum_replicated` |

## 5. Trust boundary — every `#[verifier::external_body]` in the artifact

These are the axioms the whole paper's correctness rests on. Jun's
audit must confirm each faithfully mirrors either a Python behavior or
a well-known mathematical fact.

### Mathematical axioms (well-known facts)

| Axiom | Where | What it says |
|-------|-------|--------------|
| AXIOM_M1 | Ex01 col, Ex02-04 | matmul splits over out-dim of weight |
| AXIOM_M2 | Ex01 row, Ex02 | matmul splits over in-dim with sum |
| AXIOM_S1 | Ex02 | silu × mul commutes with dim-1 concat |
| AXIOM_ATTN_HEAD_LOCAL | Ex03 | attention commutes with head-shard |
| AXIOM_RI1 (length + content) | Ex04 | repeat_interleave semantics |
| AXIOM_SOFTMAX_SUM | Ex05 | softmax sums to 1 |

### Python-correspondence axioms (Verus-Python bridge)

| Axiom | Where | What it says |
|-------|-------|--------------|
| `axiom_broadcast_produces_replicated` | `axiom_base.rs` | dist.broadcast establishes Replicated |
| `axiom_all_reduce_sum_replicated` | `axiom_base.rs`, Ex06_ep/lean | dist.all_reduce(SUM) produces Replicated output |
| `axiom_all_reduce_produces_replicated` | Ex01 row | dist.all_reduce(SUM) on tp_group produces Replicated output — the Python↔Verus bridge for RowParallelLinear |
| `axiom_all_to_all_count_symmetric` | Ex06_ep_pure | send/recv counts negotiated by pre-dispatch all_to_all_single are symmetric |
| `axiom_tp_row_parallel_makes_replicated` | `lean_equiv_hybrid_dp1.rs` | Ex04 R4 — TP row-parallel all-reduce establishes replicated attention output |
| `axiom_lean_refines_spec` | `lean_equiv_hybrid_dp1.rs`, `composition_theorem.rs` | Ex06_ep/lean L6 — lean_forward refines MoE_forward_spec |
| `axiom_hybrid_refines_spec` | `lean_equiv_hybrid_dp1.rs`, `composition_theorem.rs` | Ex07 H6 — hybrid_forward refines MoE_forward_spec |
| `axiom_naive_refines_spec` | `naive_equiv_fused_moe.rs` | Ex05 E1 — NaiveSparseMoE refines moe_spec_pointwise |
| `axiom_permuted_refines_spec` | `naive_equiv_fused_moe.rs`, `composition_theorem.rs` | Ex05 E2 — PermutedSparseMoE refines moe_spec_pointwise |
| `axiom_fused_refines_spec` | `naive_equiv_fused_moe.rs`, `composition_theorem.rs` | Ex09 F4 — fused Triton kernel refines moe_spec_pointwise |
| `axiom_variant_conserves_records` | `composition_theorem.rs` §7.1 | any variant preserves records_out == records_in * top_k |
| `axiom_variant_schedule_terminates` | `composition_theorem.rs` §7.2 | any variant's collective schedule terminates deadlock-free |
| `axiom_variant_unique_writer` | `composition_theorem.rs` §7.3 | any variant maintains the unique-writer invariant |

### Per-component stubs (deferred proofs)

These are `external_body` in per-component files because their full
mechanization is future work. Each is scoped to a specific claim whose
proof structure is documented inline.

| Stub | Where | What's deferred |
|------|-------|-----------------|
| `c4_forward_correctness_general` | Ex01 col | Parameterized-over-tp_size version of C4 |
| `r4_forward_correctness_general_stub` | Ex01 row | Parameterized-over-tp_size version of R4 |
| `t3_block_correctness_stub` | Ex02 | Full MLP-TP block correctness |
| `q4_three_way_split_stub` | Ex03 | Three-way QKV split proof |
| `a2_block_correctness_stub` | Ex03 | Full MHA-TP block correctness |
| `r2_replica_siblings_identical_kv_stub` | Ex04 | Replica-sibling attention equality |
| `r3_block_correctness_stub` | Ex04 | Full GQA-TP block correctness |
| `e1_naive_refines_spec_stub` | Ex05 | Naive MoE per-token refinement |
| `e2_permuted_refines_naive_stub` | Ex05 | Permuted-vs-naive equivalence |
| `ep5_round_trip_stub` | Ex06_ep_pure | Dispatch-combine round-trip |
| `ep6_deadlock_free_stub` | Ex06_ep_pure | Deadlock-freedom (structural) |
| `ep7_routing_correctness_stub` | Ex06_ep_pure | Full dispatch-based routing correctness |
| `l3_partial_output_zero_outside` | Ex06_ep/lean | Zero-outside-contributing (index_add_ postcondition) |
| `l4_sum_of_partials_stub` | Ex06_ep/lean | Sum-of-partials-equals-MoE_spec |
| `l6_refines_moe_spec_stub` | Ex06_ep/lean | Full lean forward refinement |
| `lean_equiv_dispatch_composition_stub` | Ex06_ep/lean | Redundant — subsumed by verus/lean_equiv_hybrid_dp1.rs |
| `h1_attn_output_tp_replicated_stub` | Ex07 | Inherited from Ex04 R4 |
| `h3_ep_dispatch_symmetric_stub` | Ex07 | Inherited from Ex06 EP3 |
| `h4_moe_out_tp_replicated_stub` | Ex07 | All-gather postcondition |
| `h5_subgroup_deadlock_free_stub` | Ex07 | Sub-group deadlock-freedom (structural) |
| `h6_block_correctness_stub` | Ex07 | Full block correctness |
| `f2_postcondition_determines_output_stub` | Ex09 | Fused kernel determinism up to approx_eq |
| `f4_fused_matches_python_loop_stub` | Ex09 | Full fused-vs-Python-loop equivalence |

**Note on stubs:** the composition-theorem files (Tier 3) invoke these
stubs as axioms and derive real Contribution 2 claims from them. Jun's
priority-1 audit is confirming that each stub's ensures-clause is a
faithful abstraction of what the corresponding Python code guarantees
— once he signs off on the stubs, the Tier 3 composition theorems are
sound.

## 6. Audit priority order (Jun should read in this sequence)

For maximum paper-impact-per-hour of audit time:

### Priority 1 — the load-bearing 5 lines (30 min)

Read `verus/composition_theorem.rs` §3
(`theorem_shared_spec_implies_equiv`). This is 5 lines of proof body
plus 3 approx_eq lemmas (refl / sym / trans). If this is sound, every
Contribution 2 theorem composes from it. **Highest audit-leverage per
line in the whole artifact.**

### Priority 2 — the 12 refinement axioms (2 hours)

Read every `#[verifier::external_body]` in
`verus/composition_theorem.rs`, `verus/lean_equiv_hybrid_dp1.rs`, and
`verus/naive_equiv_fused_moe.rs`. For each, open the corresponding
Python file (`bootcamp/exNN/*.py`) and confirm the axiom's
ensures-clause matches the Python's actual behavior. This is the
Python-correspondence audit that the paper's methodology section
argues is the human-audit's central function.

### Priority 3 — the anecdote-driven proof (30 min)

Read
[`bootcamp/ex02_mlp_tp/verification/anecdote.txt`](../bootcamp/ex02_mlp_tp/verification/anecdote.txt)
alongside
[`bootcamp/ex02_mlp_tp/verification/mlp_tp.rs`](../bootcamp/ex02_mlp_tp/verification/mlp_tp.rs).
This is the concrete example the paper's Contribution 3 uses. Confirm
the anecdote accurately describes the two audit cycles.

### Priority 4 — safety theorems (1 hour)

Read `verus/composition_theorem.rs` §7 (S1-S3). Confirm the three
safety-axioms match their per-component sources. In particular:
- S1's `axiom_variant_conserves_records` should compose Ex05 RT1
  (moe_baseline.rs), Ex06_ep_pure EP4 (ep_pure.rs), and Ex09 F1
  (fused_moe.rs). Confirm each of those actually states what the
  axiom claims.
- S2's deadlock-freedom is a structural property; confirm the
  per-component `ep6_deadlock_free_stub` and `h5_subgroup_deadlock_free_stub`
  say what the paper claims.
- S3's unique-writer is subtle — the claim is that atomic-free scatter
  (index_add_ into per-rank buffers) is well-defined without race
  conditions. Confirm this is faithful to Python semantics.

### Priority 5 — per-component proofs (4-6 hours)

For each `bootcamp/exNN_*/verification/*.rs`, read the `PROPERTIES.md`
first, then the `.rs`. Audit questions per file:
1. Do the properties in `PROPERTIES.md` correspond to the exercise's
   actual semantics? Read `solution.py` (or `reference_lean.py` for
   Ex06_ep) alongside.
2. Does the `.rs` file's stated theorems match `PROPERTIES.md`?
3. Are the axioms sound?
4. Are the theorem bodies free of `admit()`?

Ex01-Ex05 are structurally similar (same TP + MoE patterns). Ex06 and
Ex07 introduce distributed collectives. Ex09 has the fused-kernel
algorithmic contract.

## 7. What the paper cites, precisely

The paper's §Contribution 2 section refers to these lemmas by name:

- Meta-theorem: `theorem_shared_spec_implies_equiv` (composition_theorem.rs)
- Schedule swap: `theorem_lean_equiv_hybrid_dp1` (lean_equiv_hybrid_dp1.rs)
  and `theorem_lean_equiv_hybrid` (composition_theorem.rs, simplified form)
- Kernel swap: `theorem_permuted_equiv_fused` (naive_equiv_fused_moe.rs)
- 2×2 block grid: `corollary_block_variants_equivalent` (composition_theorem.rs)
- Token conservation: `theorem_token_conservation` (composition_theorem.rs)
- Deadlock-freedom: `theorem_deadlock_free` (composition_theorem.rs)
- Unique-writer: `theorem_unique_writer` (composition_theorem.rs)

The paper's §Methodology section refers to these files:
- Per-component proof template: `bootcamp/ex01_linear_tp/verification/`
- Audit-cycle anecdote: `bootcamp/ex02_mlp_tp/verification/anecdote.txt`
- Shared framework: `verus/axiom_base.rs`
- Composition-theorem home: `verus/composition_theorem.rs`

## 8. Known limitations Jun should be aware of

1. **`approx_eq` is shape-only at the Verus level.** Its actual numerical
   content is not modeled; downstream lemmas treat it opaquely.
   Strengthening to a fully-numerical predicate is future work.
2. **Every per-component "block correctness" (T3/A2/R3/H6/EP7/L6) is stubbed.**
   The Tier 3 composition theorems invoke these as axioms. Full
   mechanization is future work.
3. **The Triton kernel's DSL semantics is modeled and structurally verified
   (`fused_kernel_dsl.rs`, K1a/K1b/K1c), but PTX/SASS compilation and A100
   hardware execution are trusted below the DSL level.** The kernel's
   correspondence to its Verus DSL model is established empirically via
   `bootcamp/tests/test_ex09_fused_moe.py` (8 tests: fp32/bf16 × uniform/skewed × small/Qwen3-scale).
   This is a strictly smaller trust surface than before: F4 (the fused-vs-Python-loop
   equivalence) is no longer purely an external axiom; it decomposes via K1 + K2 +
   AXIOM_MATMUL_SPLITS_OVER_K, with K2 (K-reduce correctness) being the one
   remaining structural stub inside fused_kernel_dsl.rs.
4. **Every per-component `c4_forward_correctness_general` / `r4_forward_correctness_general` is stubbed.**
   The parameterized-over-tp_size versions. Concrete tp_size=2 versions
   are proved.

The paper's threats-to-validity section lists these limitations
explicitly.

## 9. Running everything

```sh
# Set up.
export PATH=/gpfs/users/weiz/verus/verus-x86-linux:$PATH

# Verify per-component proofs.
cd /gpfs/users/weiz/workspace/personal/qwen-moe
for f in bootcamp/ex01_linear_tp/verification/column_parallel.rs \
         bootcamp/ex01_linear_tp/verification/row_parallel.rs \
         bootcamp/ex02_mlp_tp/verification/mlp_tp.rs \
         bootcamp/ex03_mha_tp/verification/mha_tp.rs \
         bootcamp/ex04_gqa_tp/verification/gqa_tp.rs \
         bootcamp/ex05_moe_baseline/verification/moe_baseline.rs \
         bootcamp/ex06_ep/verification/lean.rs \
         bootcamp/ex06_ep_pure/verification/ep_pure.rs \
         bootcamp/ex07_tp_ep_hybrid/verification/hybrid.rs \
         bootcamp/ex09_fused_moe/verification/fused_moe.rs \
         bootcamp/ex09_fused_moe/verification/fused_kernel_dsl.rs; do
    dir=$(dirname "$f"); file=$(basename "$f")
    (cd "$dir" && verus "$file" 2>&1 | grep "verification results")
done

# Verify Tier 2 + Tier 3.
verus --crate-type=lib verus/axiom_base.rs
verus --crate-type=lib verus/lean_equiv_hybrid_dp1.rs
verus --crate-type=lib verus/naive_equiv_fused_moe.rs
verus --crate-type=lib verus/composition_theorem.rs
```

Expected output: every command reports `verification results:: N verified, 0 errors`.

Total wall-clock time: ~2 minutes on the shared pod.

## 10. Audit sign-off form

For each file Jun audits, please record in a text log:

```
File: bootcamp/exNN_XXX/verification/YYY.rs (or verus/YYY.rs)
Date audited:
Verified count matches this doc: [ Y / N ]
Axioms sound: [ Y / N — if N, list issues ]
Contracts non-decorative: [ Y / N — if N, list which ]
Proof body honest (no admit()): [ Y / N ]
Correspondence to Python solid: [ Y / N — if N, list issues ]

Comments:
```

The audit log itself is the paper's methodology evidence. Please keep
it in `verus/audit_log.md` (or a filename Jun prefers) — the paper's
Contribution 3 section will cite this log as the concrete "audit
trail" that turns the AI-drafted / human-audited workflow claim into
verifiable practice.
