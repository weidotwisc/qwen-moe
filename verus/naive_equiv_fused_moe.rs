// verus/naive_equiv_fused_moe.rs
//
// The paper's kernel-swap composition theorem: equivalence between the
// per-expert Python implementations (Ex05) and the fused Triton grouped-GEMM
// kernel (Ex09).
//
// Under the shared precondition that routing is consistent, the three
// implementations refine the same exact semantic output. Their pairwise
// equivalence is derived with the shared theorem in composition_core.rs.
//
// Run with:
//   verus --crate-type=lib verus/naive_equiv_fused_moe.rs

use vstd::prelude::*;

#[path = "composition_core.rs"]
mod composition_core;
use composition_core::{Tensor, semantic_eq, theorem_shared_spec_implies_equiv};

verus! {

pub type TokenId = nat;
pub type ExpertId = nat;

// =====================================================================
// §1 — Shared semantic spec and implementation outputs.
// =====================================================================

/// Exact abstract output of the reference MoE semantics.
pub uninterp spec fn moe_spec_pointwise(x: Tensor) -> Tensor;

/// Ex05 NaiveSparseMoE's forward output (per-token per-expert loop).
pub uninterp spec fn naive_forward(x: Tensor) -> Tensor;

/// Ex05 PermutedSparseMoE's forward output (argsort + grouped + scatter).
pub uninterp spec fn permuted_forward(x: Tensor) -> Tensor;

/// Ex09 fused Triton kernel's forward output (grouped-GEMM launch).
pub uninterp spec fn fused_forward(x: Tensor) -> Tensor;

// =====================================================================
// §2 — Routing-consistency precondition.
// =====================================================================

/// The input has valid top-k routing, permutation, and expert offsets.
/// Its unfolded form is supplied by Ex05's routing invariants.
pub uninterp spec fn RoutingConsistent(x: Tensor) -> bool;

// =====================================================================
// §3 — Refinement contracts required from the component proofs.
// =====================================================================

/// Contract expected from Ex05 E1.
pub open spec fn naive_refines_spec(x: Tensor) -> bool {
    RoutingConsistent(x)
        ==> semantic_eq(naive_forward(x), moe_spec_pointwise(x))
}

/// Contract expected from Ex05 E2.
pub open spec fn permuted_refines_spec(x: Tensor) -> bool {
    RoutingConsistent(x)
        ==> semantic_eq(permuted_forward(x), moe_spec_pointwise(x))
}

/// Contract expected from Ex09 F4.
pub open spec fn fused_refines_spec(x: Tensor) -> bool {
    RoutingConsistent(x)
        ==> semantic_eq(fused_forward(x), moe_spec_pointwise(x))
}

// =====================================================================
// §4 — Instances of the shared composition theorem.
// =====================================================================

pub proof fn theorem_naive_equiv_fused(x: Tensor)
    requires
        RoutingConsistent(x),
        naive_refines_spec(x),
        fused_refines_spec(x),
    ensures semantic_eq(naive_forward(x), fused_forward(x)),
{
    theorem_shared_spec_implies_equiv(
        naive_forward(x), fused_forward(x), moe_spec_pointwise(x),
    );
}

pub proof fn theorem_permuted_equiv_fused(x: Tensor)
    requires
        RoutingConsistent(x),
        permuted_refines_spec(x),
        fused_refines_spec(x),
    ensures semantic_eq(permuted_forward(x), fused_forward(x)),
{
    theorem_shared_spec_implies_equiv(
        permuted_forward(x), fused_forward(x), moe_spec_pointwise(x),
    );
}

pub proof fn corollary_naive_equiv_permuted(x: Tensor)
    requires
        RoutingConsistent(x),
        naive_refines_spec(x),
        permuted_refines_spec(x),
    ensures semantic_eq(naive_forward(x), permuted_forward(x)),
{
    theorem_shared_spec_implies_equiv(
        naive_forward(x), permuted_forward(x), moe_spec_pointwise(x),
    );
}

// =====================================================================
// §5 — Smoke tests.
// =====================================================================

pub proof fn smoke_test_theorem_1(x: Tensor)
    requires
        RoutingConsistent(x),
        naive_refines_spec(x),
        fused_refines_spec(x),
    ensures semantic_eq(naive_forward(x), fused_forward(x)),
{
    theorem_naive_equiv_fused(x);
}

pub proof fn smoke_test_theorem_2(x: Tensor)
    requires
        RoutingConsistent(x),
        permuted_refines_spec(x),
        fused_refines_spec(x),
    ensures semantic_eq(permuted_forward(x), fused_forward(x)),
{
    theorem_permuted_equiv_fused(x);
}

} // verus!
