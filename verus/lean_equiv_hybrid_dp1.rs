// verus/lean_equiv_hybrid_dp1.rs
//
// The paper's HEADLINE COMPOSITION THEOREM.
//
// Under the topology constraint tp × dp = ep = world_size with dp = 1
// (i.e., tp_size == ep_size == world_size == every rank is in the single
// tp_group and the single ep_group), the following two schedules produce
// semantically equal outputs on every rank:
//
//   (A) bootcamp/ex06_ep/reference_lean.py       — lean, single all_reduce
//   (B) bootcamp/ex07_tp_ep_hybrid/solution.py    — TP × DP × EP hybrid,
//                                                    dispatch-based
//
// This IS the composition-theorem claim that Contribution 2 of the paper
// argues. Its proof is a direct application of the shared composition
// theorem to the two individual refinement facts:
//
//   - Ex06_ep lean's L6:   lean_forward(x)   refines MoE_forward_spec(x)
//   - Ex07's H6:           hybrid_forward(x) refines MoE_forward_spec(x)
//
// Both hold under the `ReplicatedInput(x, x_id, ep_group)` precondition. The preceding
// TP row-parallel stage must supply this as its proved postcondition.
//
// The per-component refinement facts are explicit premises of this theorem.
// Their full mechanization remains a separate proof obligation.
//
// Run with:
//   verus --crate-type=lib verus/lean_equiv_hybrid_dp1.rs

use vstd::prelude::*;

#[path = "composition_core.rs"]
mod composition_core;
use composition_core::{Tensor, semantic_eq, theorem_shared_spec_implies_equiv};

verus! {

// =====================================================================
// §1 — Distributed types used by this instance.
// =====================================================================

pub type Rank = nat;
pub type ExpertId = nat;
pub type Group = Set<Rank>;

/// The topology required by this schedule equivalence:
/// tp_size * dp_size == ep_size == world_size and dp_size == 1. Under this
/// specialization both the TP and EP group contain exactly the world ranks.
pub open spec fn valid_dp1_topology(
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
) -> bool {
    &&& world_size > 0
    &&& tp_size > 0
    &&& ep_size > 0
    &&& dp_size == 1
    &&& tp_size * dp_size == ep_size
    &&& ep_size == world_size
    &&& forall|r: Rank| #![auto]
        tp_group.contains(r) == (r < world_size)
    &&& forall|r: Rank| #![auto]
        ep_group.contains(r) == (r < world_size)
}

// =====================================================================
// §2 — Replication predicate.
// =====================================================================

pub uninterp spec fn tensor_on(tensor_id: nat, rank: Rank) -> Tensor;

/// Every rank in a non-empty group observes the exact input value `x` under
/// identity `tensor_id`. This both links the id to the theorem argument and
/// implies pairwise replication.
pub open spec fn ReplicatedInput(x: Tensor, tensor_id: nat, g: Group) -> bool {
    &&& exists|r: Rank| g.contains(r)
    &&& forall|r: Rank| g.contains(r) ==> tensor_on(tensor_id, r) == x
}

// =====================================================================
// §3 — The shared semantic spec function (MoE_forward_spec).
// =====================================================================

/// Uninterpreted per-token MoE semantic function. Both schedule variants
/// refine its exact abstract output.
pub uninterp spec fn moe_forward_spec(x: Tensor) -> Tensor;

// =====================================================================
// §4 — Refinement contracts required from the component proofs.
// =====================================================================

/// Lean variant's forward output for a given input tensor.
pub uninterp spec fn lean_forward(x: Tensor) -> Tensor;

/// Hybrid variant's forward output for a given input tensor.
pub uninterp spec fn hybrid_forward(x: Tensor) -> Tensor;

/// Contract expected from Ex06 L6: replicated input implies that the lean
/// implementation refines the shared semantic output.
pub open spec fn lean_refines_spec(
    x: Tensor, x_id: nat, ep_group: Group,
) -> bool {
    ReplicatedInput(x, x_id, ep_group)
        ==> semantic_eq(lean_forward(x), moe_forward_spec(x))
}

/// Contract expected from Ex07 H6: replicated input implies that the hybrid
/// implementation refines the same semantic output.
pub open spec fn hybrid_refines_spec(
    x: Tensor, x_id: nat, ep_group: Group,
) -> bool {
    ReplicatedInput(x, x_id, ep_group)
        ==> semantic_eq(hybrid_forward(x), moe_forward_spec(x))
}

// =====================================================================
// §5 — The composition theorem.
// =====================================================================

/// THEOREM (paper's Contribution 2 headline):
///
/// Under the topology constraint dp = 1, tp = ep = world_size:
/// if the preceding TP stage has established that the MoE input is
/// Replicated on tp_group, then group equality transfers that fact to
/// ep_group and the two schedules produce semantically equal outputs.
pub proof fn theorem_lean_equiv_hybrid_dp1(
    x: Tensor,
    x_id: nat,
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
)
    requires
        valid_dp1_topology(
            world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        ),
        // Postcondition supplied by the preceding TP row-parallel stage.
        ReplicatedInput(x, x_id, tp_group),
        // These are the exact postconditions that Ex06 L6 and Ex07 H6
        // must establish; they are obligations, not trusted axioms here.
        lean_refines_spec(x, x_id, ep_group),
        hybrid_refines_spec(x, x_id, ep_group),
    ensures
        semantic_eq(lean_forward(x), hybrid_forward(x)),
{
    // Step 1: DP=1 makes both groups cover exactly the world ranks, so the
    // TP replication postcondition transfers to the EP group.
    assert(ep_group.contains(0nat));
    assert(exists|r: Rank| ep_group.contains(r)) by {
        assert(ep_group.contains(0nat));
    }
    assert forall|r: Rank| ep_group.contains(r)
        implies tensor_on(x_id, r) == x by {
        assert(r < world_size);
        assert(tp_group.contains(r));
    }
    assert(ReplicatedInput(x, x_id, ep_group));

    // Step 2: instantiate the two caller-supplied refinement contracts.
    assert(semantic_eq(lean_forward(x), moe_forward_spec(x)));
    assert(semantic_eq(hybrid_forward(x), moe_forward_spec(x)));

    // Step 3: invoke the shared composition theorem.
    theorem_shared_spec_implies_equiv(
        lean_forward(x), hybrid_forward(x), moe_forward_spec(x),
    );
}

} // verus!
