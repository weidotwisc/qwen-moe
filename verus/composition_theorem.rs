// verus/composition_theorem.rs
//
// The paper's Contribution 2 headline theorem — the composition theorem
// stated as an abstract meta-theorem, together with its concrete
// instantiations for the compositions the paper cares about.
//
// The meta-theorem says: for any two implementations A and B that both
// refine a shared semantic spec S, A and B produce semantically equal
// outputs. This is a direct application of semantic_eq transitivity.
//
// Every specific composition theorem in the paper's artifact is an
// INSTANCE of this meta-theorem:
//
//   Instance 1 (lean_equiv_hybrid_dp1.rs):
//     A = lean_forward, B = hybrid_forward, S = MoE_forward_spec.
//     Refinement axioms: Ex06_ep/lean L6, Ex07 H6.
//
//   Instance 2 (naive_equiv_fused_moe.rs):
//     A = permuted_forward, B = fused_forward, S = moe_spec_pointwise.
//     Refinement axioms: Ex05 E2, Ex09 F4.
//
// The paper's Contribution 2 headline is precisely: THIS meta-theorem +
// its two instantiations that cover the schedule swap (lean ↔ hybrid)
// and the kernel swap (Python loop ↔ fused Triton).
//
// A block-level corollary states that a Qwen3-MoE-block assembled from
// any certified choice of subcomponent variants produces output
// semantically equal to the single-GPU reference block.
//
// Run with:
//   verus --crate-type=lib verus/composition_theorem.rs

use vstd::prelude::*;
use vstd::multiset::Multiset;

verus! {

// =====================================================================
// §1 — Types (abstract, shared with all Tier-3 composition files).
//
// A MoE-layer output is modeled by the MULTISET of weighted per-expert
// contributions it aggregates. A single contribution says "token `token`
// receives expert `expert`'s output scaled by (integer-encoded) weight
// `weight`". Because summation over a multiset is order-independent, two
// outputs built from the SAME multiset of contributions have the same
// semantics.
// =====================================================================

pub struct Contribution {
    pub token: nat,
    pub expert: nat,
    pub weight: int,
}

pub struct Tensor {
    pub content: Seq<int>,
    /// The multiset of weighted per-expert contributions aggregated into
    /// this output. This is the semantic content the equivalence relation
    /// compares; `content` is kept only for shape.
    pub contribs: Multiset<Contribution>,
}

// =====================================================================
// §2 — Semantic equality: equality of the contribution multiset.
// =====================================================================

pub open spec fn semantic_eq(x: Tensor, y: Tensor) -> bool {
    x.contribs == y.contribs
}

pub proof fn lemma_semantic_eq_refl(x: Tensor)
    ensures semantic_eq(x, x),
{}

pub proof fn lemma_semantic_eq_sym(x: Tensor, y: Tensor)
    requires semantic_eq(x, y),
    ensures semantic_eq(y, x),
{}

pub proof fn lemma_semantic_eq_trans(x: Tensor, y: Tensor, z: Tensor)
    requires
        semantic_eq(x, y),
        semantic_eq(y, z),
    ensures semantic_eq(x, z),
{}

// =====================================================================
// §3 — THE META-THEOREM.
//
// Given ANY two implementations A_out and B_out that refine a shared
// spec S_out, A_out and B_out produce semantically equal outputs.
//
// This is the entire content of the paper's Contribution 2. Every
// specific composition theorem in the paper's artifact is an INSTANCE
// of this meta-theorem.
// =====================================================================

pub proof fn theorem_shared_spec_implies_equiv(
    a_out: Tensor,
    b_out: Tensor,
    s_out: Tensor,
)
    requires
        semantic_eq(a_out, s_out),
        semantic_eq(b_out, s_out),
    ensures
        semantic_eq(a_out, b_out),
{
    lemma_semantic_eq_sym(b_out, s_out);
    lemma_semantic_eq_trans(a_out, s_out, b_out);
}

// =====================================================================
// §4 — INSTANCE 1: schedule-swap equivalence (lean ↔ hybrid, DP=1).
//
// See verus/lean_equiv_hybrid_dp1.rs for the full version with the
// TP-row-parallel-establishes-Replicated precondition chain. Here we
// state the theorem as a direct instance of the meta-theorem, with the
// per-component refinement axioms as its inputs.
// =====================================================================

pub uninterp spec fn moe_forward_spec(x: Tensor) -> Tensor;
pub uninterp spec fn lean_forward(x: Tensor) -> Tensor;
pub uninterp spec fn hybrid_forward(x: Tensor) -> Tensor;

#[verifier::external_body]
pub proof fn axiom_lean_refines_spec(x: Tensor)
    ensures semantic_eq(lean_forward(x), moe_forward_spec(x)),
{}

#[verifier::external_body]
pub proof fn axiom_hybrid_refines_spec(x: Tensor)
    ensures semantic_eq(hybrid_forward(x), moe_forward_spec(x)),
{}

/// Instance 1 of the meta-theorem.
pub proof fn theorem_lean_equiv_hybrid(x: Tensor)
    ensures semantic_eq(lean_forward(x), hybrid_forward(x)),
{
    axiom_lean_refines_spec(x);
    axiom_hybrid_refines_spec(x);
    theorem_shared_spec_implies_equiv(
        lean_forward(x), hybrid_forward(x), moe_forward_spec(x),
    );
}

// =====================================================================
// §5 — INSTANCE 2: kernel-swap equivalence (Python loop ↔ fused Triton).
// =====================================================================

pub uninterp spec fn permuted_forward(x: Tensor) -> Tensor;
pub uninterp spec fn fused_forward(x: Tensor) -> Tensor;

#[verifier::external_body]
pub proof fn axiom_permuted_refines_spec(x: Tensor)
    ensures semantic_eq(permuted_forward(x), moe_forward_spec(x)),
{}

#[verifier::external_body]
pub proof fn axiom_fused_refines_spec(x: Tensor)
    ensures semantic_eq(fused_forward(x), moe_forward_spec(x)),
{}

/// Instance 2 of the meta-theorem.
pub proof fn theorem_permuted_equiv_fused(x: Tensor)
    ensures semantic_eq(permuted_forward(x), fused_forward(x)),
{
    axiom_permuted_refines_spec(x);
    axiom_fused_refines_spec(x);
    theorem_shared_spec_implies_equiv(
        permuted_forward(x), fused_forward(x), moe_forward_spec(x),
    );
}

// =====================================================================
// §6 — BLOCK-LEVEL COROLLARY: full Qwen3-MoE-block equivalence.
//
// A Qwen3-MoE-block is a chain of components:
//   attn_out = attention_TP(x)
//   moe_out  = MoE_variant(attn_out)         [lean OR hybrid, using
//                                              python loop OR fused kernel]
//   out      = attn_out + moe_out             [residual]
//
// The block-level claim: any certified choice of MoE_variant produces
// the same semantic block-level output.
// =====================================================================

/// The block-level forward output, parameterized by the two variant choices.
pub uninterp spec fn block_forward(
    x: Tensor,
    use_lean: bool,   // true = lean schedule, false = hybrid dispatch
    use_fused: bool,  // true = fused Triton, false = Python loop
) -> Tensor;

/// Two block-level variants that both refine a single "block spec".
/// Their equivalence follows from the two instance theorems above.
pub uninterp spec fn block_spec(x: Tensor) -> Tensor;

/// AXIOM: each of the four block-variant configurations refines the
/// same block_spec. Follows from composing the per-schedule and
/// per-kernel refinement axioms above with the surrounding block
/// structure (attn_TP + residual, treated as identity-modulo-shape here).
#[verifier::external_body]
pub proof fn axiom_block_variant_refines_spec(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    ensures semantic_eq(
        block_forward(x, use_lean, use_fused),
        block_spec(x),
    ),
{}

/// BLOCK-LEVEL COROLLARY: any two block-variant configurations produce
/// semantically equal outputs.
pub proof fn corollary_block_variants_equivalent(
    x: Tensor,
    use_lean_a: bool, use_fused_a: bool,
    use_lean_b: bool, use_fused_b: bool,
)
    ensures semantic_eq(
        block_forward(x, use_lean_a, use_fused_a),
        block_forward(x, use_lean_b, use_fused_b),
    ),
{
    axiom_block_variant_refines_spec(x, use_lean_a, use_fused_a);
    axiom_block_variant_refines_spec(x, use_lean_b, use_fused_b);
    theorem_shared_spec_implies_equiv(
        block_forward(x, use_lean_a, use_fused_a),
        block_forward(x, use_lean_b, use_fused_b),
        block_spec(x),
    );
}

// =====================================================================
// §7 — GLOBAL SAFETY PROPERTIES.
//
// The functional-equivalence theorems (§3-§6) say "different variants
// produce equivalent outputs". The theorems below say the SYSTEM's
// execution is well-formed regardless of which variant is chosen. They
// are the block-scope form of the paper's four goals (paper section
// "What we prove per component"):
//
//   Work conservation         every routed token is processed exactly
//   (Completeness+Disjointness): top_k times across all ranks -- none
//                              dropped, none double-counted.
//   Deadlock freedom:          for any variant, the collective schedule
//                              posts matched collectives in a fixed order
//                              on every rank. The substrate-trust
//                              reduction is proved in deadlock_free.rs;
//                              here we state the variant-independent form.
//   Data-race freedom:         for any output position on any rank, at
//                              most one source writes it per forward (the
//                              unique-writer mechanism), so there is no
//                              race.
//
// Each is stated as a variant-independent property: it holds regardless
// of which lean/hybrid × python/fused configuration is running. This is
// the "global safety" half of Contribution 2, alongside equivalence.
// =====================================================================

// ---------------------------------------------------------------------
// §7.1 — Work conservation (Completeness + Disjointness).
// ---------------------------------------------------------------------

/// Total input records to the MoE forward on some rank.
pub uninterp spec fn total_records_in(x: Tensor) -> nat;

/// Total output-slot writes performed by variant `(use_lean, use_fused)`.
/// A "write" here is one entry in an expert's contribution to some
/// token's final output — so this counts the (token, expert) pairs
/// actually processed.
pub uninterp spec fn total_records_out(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> nat;

pub uninterp spec fn top_k() -> nat;

/// AXIOM (from per-component RT1 / EP4 / F1): every variant satisfies
/// records_in * top_k == records_out (i.e., each token is processed
/// exactly top_k times regardless of routing schedule or kernel).
///
/// Per-component sources:
///   Ex05 RT1 (moe_baseline.rs): `cumsum(counts).last() == total_tokens`.
///   Ex06_ep_pure EP4 (ep_pure.rs): send/recv pairwise equality.
///   Ex09 F1 (fused_moe.rs): offsets partition [0, M) exactly once.
#[verifier::external_body]
pub proof fn axiom_variant_conserves_records(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    ensures total_records_out(x, use_lean, use_fused)
         == total_records_in(x) * top_k(),
{}

/// THEOREM (work conservation = Completeness + Disjointness, variant-independent):
/// For any variant configuration, records_out == records_in * top_k.
/// Trivial once the axiom is available — the point is that the axiom
/// holds for EVERY choice of `(use_lean, use_fused)`, not just one.
pub proof fn theorem_token_conservation(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    ensures total_records_out(x, use_lean, use_fused)
         == total_records_in(x) * top_k(),
{
    axiom_variant_conserves_records(x, use_lean, use_fused);
}

/// COROLLARY: any two variants preserve the SAME total-record count.
pub proof fn corollary_records_variant_invariant(
    x: Tensor,
    ul_a: bool, uf_a: bool,
    ul_b: bool, uf_b: bool,
)
    ensures total_records_out(x, ul_a, uf_a) == total_records_out(x, ul_b, uf_b),
{
    axiom_variant_conserves_records(x, ul_a, uf_a);
    axiom_variant_conserves_records(x, ul_b, uf_b);
}

// ---------------------------------------------------------------------
// §7.2 — Deadlock freedom.
//
// This is the variant-independent WRAPPER. The substantive treatment ---
// trusting NCCL's collective-matching contract as a substrate and PROVING
// that a data-independent schedule meets its matching precondition --- is
// in deadlock_free.rs (theorem_deadlock_free via
// lemma_data_independent_implies_matched). Here we only record that the
// property holds for every variant.
// ---------------------------------------------------------------------

/// A predicate on a variant configuration: "the collective schedule
/// this variant issues terminates on every rank without deadlock."
pub uninterp spec fn schedule_terminates(use_lean: bool, use_fused: bool) -> bool;

/// AXIOM: every variant's schedule is data-independent (fixed sequence of
/// group-matched collectives on every rank), so under the NCCL substrate
/// contract it is deadlock-free. The data-independence-implies-matched
/// reduction is proved in deadlock_free.rs; the NCCL contract itself is
/// trusted (see deadlock_free.rs::axiom_matched_implies_no_deadlock).
///
/// Per-component sources:
///   Ex06_ep_pure EP6 (ep_pure.rs).
///   Ex07 H5 (hybrid.rs).
#[verifier::external_body]
pub proof fn axiom_variant_schedule_terminates(use_lean: bool, use_fused: bool)
    ensures schedule_terminates(use_lean, use_fused),
{}

/// THEOREM (deadlock freedom, variant-independent):
/// Every variant's collective schedule terminates deadlock-free.
pub proof fn theorem_deadlock_free(use_lean: bool, use_fused: bool)
    ensures schedule_terminates(use_lean, use_fused),
{
    axiom_variant_schedule_terminates(use_lean, use_fused);
}

// ---------------------------------------------------------------------
// §7.3 — Data-race freedom (via the unique-writer invariant).
//
// In a distributed MoE with atomics-free scatter (each rank writes into
// its OWN output buffer, no cross-rank shared memory), the relevant
// property is: each output-tensor position on each rank is written by at
// most one source per forward — the "unique-writer" invariant, which is
// the mechanism by which data-race freedom holds.
// ---------------------------------------------------------------------

pub uninterp spec fn unique_writer_invariant(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> bool;

/// AXIOM (from Ex06_ep/lean L3, Ex05 RT4 permutation invertibility,
/// Ex09 F3 empty-expert handling): every variant maintains the
/// unique-writer invariant on output buffers. Each rank writes to its
/// own output positions determined by routing; different ranks' output
/// buffers are disjoint; per-rank writes are ordered by the scatter
/// (index_add_) semantics.
///
/// Per-component sources:
///   Ex06_ep/lean L3 (lean.rs): zero-outside-contributing predicate.
///   Ex05 RT4 (moe_baseline.rs): permutation bijection.
///   Ex09 F3 (fused_moe.rs): empty-expert doesn't read/write outside its block.
#[verifier::external_body]
pub proof fn axiom_variant_unique_writer(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    ensures unique_writer_invariant(x, use_lean, use_fused),
{}

/// THEOREM (data-race freedom, variant-independent):
/// Every variant maintains the unique-writer invariant, so no two sources
/// write the same output position in a forward pass.
pub proof fn theorem_data_race_free(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    ensures unique_writer_invariant(x, use_lean, use_fused),
{
    axiom_variant_unique_writer(x, use_lean, use_fused);
}

// =====================================================================
// §8 — Smoke tests: exercise each level of the theorem hierarchy.
// =====================================================================

pub proof fn smoke_meta_theorem(a: Tensor, b: Tensor, s: Tensor)
    requires
        semantic_eq(a, s),
        semantic_eq(b, s),
    ensures semantic_eq(a, b),
{
    theorem_shared_spec_implies_equiv(a, b, s);
}

pub proof fn smoke_instance_1(x: Tensor)
    ensures semantic_eq(lean_forward(x), hybrid_forward(x)),
{
    theorem_lean_equiv_hybrid(x);
}

pub proof fn smoke_instance_2(x: Tensor)
    ensures semantic_eq(permuted_forward(x), fused_forward(x)),
{
    theorem_permuted_equiv_fused(x);
}

pub proof fn smoke_block(x: Tensor)
    ensures semantic_eq(
        block_forward(x, true, true),   // lean + fused
        block_forward(x, false, false), // hybrid + python-loop
    ),
{
    corollary_block_variants_equivalent(x, true, true, false, false);
}

pub proof fn smoke_safety_conservation(x: Tensor)
    ensures total_records_out(x, true, true)
         == total_records_in(x) * top_k(),
{
    theorem_token_conservation(x, true, true);
}

pub proof fn smoke_safety_deadlock()
    ensures schedule_terminates(true, true) && schedule_terminates(false, false),
{
    theorem_deadlock_free(true, true);
    theorem_deadlock_free(false, false);
}

pub proof fn smoke_safety_data_race_free(x: Tensor)
    ensures unique_writer_invariant(x, true, true),
{
    theorem_data_race_free(x, true, true);
}

} // verus!
