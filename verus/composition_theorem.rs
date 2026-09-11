// verus/composition_theorem.rs
//
// The paper's Contribution 2 headline theorem — the composition theorem
// stated as an abstract meta-theorem, together with the block-level and
// safety consequences used by the paper.
//
// The meta-theorem says: for any two implementations A and B that both
// refine a shared semantic spec S, A and B produce semantically equal
// outputs. This is a direct application of semantic_eq transitivity.
//
// The specific composition theorems import and instantiate this shared
// meta-theorem directly:
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
// A block-level corollary states that any two Qwen3-MoE blocks assembled
// from certified schedule and kernel choices produce semantically equal
// outputs.
//
// Run with:
//   verus --crate-type=lib verus/composition_theorem.rs

use vstd::prelude::*;

#[path = "composition_core.rs"]
mod composition_core;
pub use composition_core::*;

#[path = "deadlock_free.rs"]
mod deadlock_model;

verus! {

// =====================================================================
// §1 — BLOCK-LEVEL COROLLARY: full Qwen3-MoE-block equivalence.
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

pub uninterp spec fn attention_forward(x: Tensor) -> Tensor;

/// MoE output for one schedule/kernel choice.
pub uninterp spec fn moe_variant_forward(
    attn_out: Tensor,
    use_lean: bool,   // true = lean schedule, false = hybrid dispatch
    use_fused: bool,  // true = fused Triton, false = Python loop
) -> Tensor;

pub uninterp spec fn residual_add(attn_out: Tensor, moe_out: Tensor) -> Tensor;

/// The block structure is explicit: attention, then MoE, then residual.
pub open spec fn block_forward(
    x: Tensor,
    use_lean: bool,
    use_fused: bool,
) -> Tensor {
    let attn_out = attention_forward(x);
    residual_add(
        attn_out,
        moe_variant_forward(attn_out, use_lean, use_fused),
    )
}

/// Obligation supplied by the schedule-swap theorem, for either kernel.
pub open spec fn schedule_swap_equiv(x: Tensor, use_fused: bool) -> bool {
    let attn_out = attention_forward(x);
    semantic_eq(
        moe_variant_forward(attn_out, true, use_fused),
        moe_variant_forward(attn_out, false, use_fused),
    )
}

/// Obligation supplied by the kernel-swap theorem, for either schedule.
pub open spec fn kernel_swap_equiv(x: Tensor, use_lean: bool) -> bool {
    let attn_out = attention_forward(x);
    semantic_eq(
        moe_variant_forward(attn_out, use_lean, true),
        moe_variant_forward(attn_out, use_lean, false),
    )
}

/// Exact equality of MoE outputs is preserved by the residual context.
pub proof fn lemma_block_context_congruence(
    x: Tensor,
    use_lean_a: bool, use_fused_a: bool,
    use_lean_b: bool, use_fused_b: bool,
)
    requires semantic_eq(
        moe_variant_forward(attention_forward(x), use_lean_a, use_fused_a),
        moe_variant_forward(attention_forward(x), use_lean_b, use_fused_b),
    ),
    ensures semantic_eq(
        block_forward(x, use_lean_a, use_fused_a),
        block_forward(x, use_lean_b, use_fused_b),
    ),
{}

/// Every variant is equivalent to the canonical lean+fused variant.
pub proof fn lemma_block_variant_equiv_canonical(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    requires
        forall|kernel: bool| schedule_swap_equiv(x, kernel),
        forall|schedule: bool| kernel_swap_equiv(x, schedule),
    ensures semantic_eq(
        block_forward(x, use_lean, use_fused),
        block_forward(x, true, true),
    ),
{
    if use_lean {
        if use_fused {
            lemma_semantic_eq_refl(block_forward(x, true, true));
        } else {
            assert(kernel_swap_equiv(x, true));
            lemma_semantic_eq_sym(
                moe_variant_forward(attention_forward(x), true, true),
                moe_variant_forward(attention_forward(x), true, false),
            );
            lemma_block_context_congruence(x, true, false, true, true);
        }
    } else {
        if use_fused {
            assert(schedule_swap_equiv(x, true));
            lemma_semantic_eq_sym(
                moe_variant_forward(attention_forward(x), true, true),
                moe_variant_forward(attention_forward(x), false, true),
            );
            lemma_block_context_congruence(x, false, true, true, true);
        } else {
            assert(kernel_swap_equiv(x, false));
            lemma_semantic_eq_sym(
                moe_variant_forward(attention_forward(x), false, true),
                moe_variant_forward(attention_forward(x), false, false),
            );
            lemma_block_context_congruence(x, false, false, false, true);

            assert(schedule_swap_equiv(x, true));
            lemma_semantic_eq_sym(
                moe_variant_forward(attention_forward(x), true, true),
                moe_variant_forward(attention_forward(x), false, true),
            );
            lemma_block_context_congruence(x, false, true, true, true);

            lemma_semantic_eq_trans(
                block_forward(x, false, false),
                block_forward(x, false, true),
                block_forward(x, true, true),
            );
        }
    }
}

/// BLOCK-LEVEL COROLLARY: any two block-variant configurations produce
/// semantically equal outputs.
pub proof fn corollary_block_variants_equivalent(
    x: Tensor,
    use_lean_a: bool, use_fused_a: bool,
    use_lean_b: bool, use_fused_b: bool,
)
    requires
        forall|kernel: bool| schedule_swap_equiv(x, kernel),
        forall|schedule: bool| kernel_swap_equiv(x, schedule),
    ensures semantic_eq(
        block_forward(x, use_lean_a, use_fused_a),
        block_forward(x, use_lean_b, use_fused_b),
    ),
{
    lemma_block_variant_equiv_canonical(x, use_lean_a, use_fused_a);
    lemma_block_variant_equiv_canonical(x, use_lean_b, use_fused_b);
    theorem_shared_spec_implies_equiv(
        block_forward(x, use_lean_a, use_fused_a),
        block_forward(x, use_lean_b, use_fused_b),
        block_forward(x, true, true),
    );
}

// =====================================================================
// §2 — GLOBAL SAFETY PROPERTIES.
//
// The functional-equivalence theorems say "different variants produce
// equivalent outputs". The theorems below say the SYSTEM's
// execution is well-formed regardless of which variant is chosen. They
// are the block-scope form of the paper's four goals (paper section
// "What we prove per component"):
//
//   Work conservation         every (token, top-k slot) routing item is
//   (Completeness+Disjointness): processed exactly once across all ranks.
//   Deadlock freedom:          for any variant, the collective schedule
//                              posts matched collectives in a fixed order
//                              on every rank. The substrate-trust
//                              reduction is proved in deadlock_free.rs;
//                              here we state the variant-independent form.
//   Data-race freedom:         top-k contributions may share an output
//                              position, but every such conflicting
//                              scatter-add is atomic.
//
// Each is stated as a variant-independent property: it holds regardless
// of which lean/hybrid × python/fused configuration is running. This is
// the "global safety" half of Contribution 2, alongside equivalence.
// =====================================================================

// ---------------------------------------------------------------------
// §2.1 — Work conservation (Completeness + Disjointness).
// ---------------------------------------------------------------------

/// A routing work item is one selected top-k slot of one input token.
/// Slots, rather than expert ids, distinguish the individual obligations.
pub struct WorkItem {
    pub token: nat,
    pub slot: nat,
}

pub uninterp spec fn input_token_count(x: Tensor) -> nat;
pub uninterp spec fn top_k(x: Tensor) -> nat;

pub open spec fn required_work_item(x: Tensor, item: WorkItem) -> bool {
    item.token < input_token_count(x) && item.slot < top_k(x)
}

/// Global number of executions of one routing item across all ranks.
pub uninterp spec fn times_processed(
    x: Tensor,
    use_lean: bool,
    use_fused: bool,
    item: WorkItem,
) -> nat;

/// Exact component-level obligation: every required item occurs once and
/// every non-required item occurs zero times.
pub open spec fn exact_work_execution(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> bool {
    forall|item: WorkItem|
        #![trigger times_processed(x, use_lean, use_fused, item)]
        times_processed(x, use_lean, use_fused, item)
            == if required_work_item(x, item) { 1nat } else { 0nat }
}

pub open spec fn work_complete(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> bool {
    forall|item: WorkItem|
        #![trigger times_processed(x, use_lean, use_fused, item)]
        required_work_item(x, item)
            ==> times_processed(x, use_lean, use_fused, item) == 1nat
}

pub open spec fn work_disjoint(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> bool {
    forall|item: WorkItem|
        #![trigger times_processed(x, use_lean, use_fused, item)]
        times_processed(x, use_lean, use_fused, item) <= 1nat
}

pub open spec fn no_spurious_work(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> bool {
    forall|item: WorkItem|
        #![trigger times_processed(x, use_lean, use_fused, item)]
        !required_work_item(x, item)
            ==> times_processed(x, use_lean, use_fused, item) == 0nat
}

/// Work conservation now proves coverage and uniqueness separately.
pub proof fn theorem_token_conservation(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    requires exact_work_execution(x, use_lean, use_fused),
    ensures
        work_complete(x, use_lean, use_fused),
        work_disjoint(x, use_lean, use_fused),
        no_spurious_work(x, use_lean, use_fused),
{
}

/// Any two correct variants execute each routing item the same number of times.
pub proof fn corollary_work_variant_invariant(
    x: Tensor,
    ul_a: bool, uf_a: bool,
    ul_b: bool, uf_b: bool,
)
    requires
        exact_work_execution(x, ul_a, uf_a),
        exact_work_execution(x, ul_b, uf_b),
    ensures forall|item: WorkItem|
        #![trigger times_processed(x, ul_a, uf_a, item)]
        times_processed(x, ul_a, uf_a, item)
            == times_processed(x, ul_b, uf_b, item),
{
}

// ---------------------------------------------------------------------
// §2.2 — Deadlock freedom.
//
// The schedule and transition-system proof live in deadlock_free.rs and are
// imported above. Kernel choice is local computation and does not change the
// collective trace; schedule choice selects the two-phase lean trace or the
// six-phase hybrid trace.
// ---------------------------------------------------------------------

pub open spec fn variant_schedule(
    world_size: nat,
    tp_size: nat,
    use_lean: bool,
    _use_fused: bool,
) -> deadlock_model::Schedule {
    if use_lean {
        deadlock_model::lean_schedule(world_size, tp_size)
    } else {
        deadlock_model::hybrid_schedule(world_size, tp_size)
    }
}

/// Every reachable state of either concrete collective schedule is
/// non-deadlocked. Runtime completion still assumes failure-free NCCL and
/// fair scheduling, as documented in deadlock_free.rs.
pub proof fn theorem_deadlock_free(
    world_size: nat,
    tp_size: nat,
    use_lean: bool,
    use_fused: bool,
    states: Seq<deadlock_model::State>,
    i: nat,
)
    requires
        world_size > 0,
        tp_size > 0,
        world_size % tp_size == 0,
        deadlock_model::execution(
            variant_schedule(world_size, tp_size, use_lean, use_fused),
            states,
        ),
        i < states.len(),
    ensures !deadlock_model::deadlocked(
        variant_schedule(world_size, tp_size, use_lean, use_fused),
        states[i as int],
    ),
{
    if use_lean {
        deadlock_model::theorem_lean_execution_deadlock_free(
            world_size, tp_size, states, i,
        );
    } else {
        deadlock_model::theorem_hybrid_execution_deadlock_free(
            world_size, tp_size, states, i,
        );
    }
}

// ---------------------------------------------------------------------
// §2.3 — Data-race freedom for atomic scatter-add.
//
// Different top-k slots for one token intentionally accumulate into the
// same output row. CUDA index_add_ implements these conflicting additions
// atomically; the order may be nondeterministic, but there is no data race.
// ---------------------------------------------------------------------

pub uninterp spec fn output_width(x: Tensor) -> nat;

pub struct OutputLocation {
    pub token: nat,
    pub feature: nat,
}

/// One scalar addition produced by one routed top-k work item.
pub struct ScatterWrite {
    pub source: WorkItem,
    /// Distinguishes repeated executions if a buggy variant processes the
    /// same work item more than once.
    pub occurrence: nat,
    pub feature: nat,
}

pub open spec fn active_scatter_write(
    x: Tensor, use_lean: bool, use_fused: bool, write: ScatterWrite,
) -> bool {
    write.occurrence < times_processed(x, use_lean, use_fused, write.source)
        && write.feature < output_width(x)
}

pub open spec fn scatter_target(write: ScatterWrite) -> OutputLocation {
    OutputLocation {
        token: write.source.token,
        feature: write.feature,
    }
}

/// Runtime property of the index_add_ implementation for this variant.
pub uninterp spec fn scatter_write_is_atomic(
    x: Tensor,
    use_lean: bool,
    use_fused: bool,
    write: ScatterWrite,
) -> bool;

/// Explicit substrate obligation: every active output accumulation is atomic.
pub open spec fn atomic_scatter_contract(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> bool {
    forall|write: ScatterWrite|
        #![trigger scatter_write_is_atomic(x, use_lean, use_fused, write)]
        active_scatter_write(x, use_lean, use_fused, write)
            ==> scatter_write_is_atomic(x, use_lean, use_fused, write)
}

/// Two distinct contributions conflict when they target the same output cell.
pub open spec fn scatter_conflict(
    x: Tensor, use_lean: bool, use_fused: bool,
    left: ScatterWrite, right: ScatterWrite,
) -> bool {
    &&& left != right
    &&& active_scatter_write(x, use_lean, use_fused, left)
    &&& active_scatter_write(x, use_lean, use_fused, right)
    &&& scatter_target(left) == scatter_target(right)
}

pub open spec fn scatter_data_race_free(
    x: Tensor, use_lean: bool, use_fused: bool,
) -> bool {
    forall|left: ScatterWrite, right: ScatterWrite| #![auto]
        scatter_conflict(x, use_lean, use_fused, left, right) ==> (
            scatter_write_is_atomic(x, use_lean, use_fused, left)
            && scatter_write_is_atomic(x, use_lean, use_fused, right)
        )
}

/// Multiple contributors to one output location are safe because every
/// conflicting read-modify-write is atomic, not because writers are unique.
pub proof fn theorem_data_race_free(
    x: Tensor, use_lean: bool, use_fused: bool,
)
    requires atomic_scatter_contract(x, use_lean, use_fused),
    ensures scatter_data_race_free(x, use_lean, use_fused),
{
}

} // verus!
