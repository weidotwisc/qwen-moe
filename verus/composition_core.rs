// Shared semantic model and composition rule for the Tier-3 proofs.

use vstd::prelude::*;

verus! {

/// A tensor's flattened exact values in row-major order.
pub struct Tensor {
    pub content: Seq<int>,
}

/// Equality in the proof's exact, mathematical output model.
pub open spec fn semantic_eq(x: Tensor, y: Tensor) -> bool {
    x == y
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

/// Any two implementations that refine the same semantic output are
/// semantically equal to each other.
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

} // verus!
