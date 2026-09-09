// verus/deadlock_free.rs
//
// Deadlock-freedom of the collective schedule, structured as a TRUST
// BOUNDARY (symmetric to how the fused Triton kernel trusts the
// Triton->PTX->hardware substrate).
//
// The claim decomposes into three layers:
//
//   (1) TRUSTED substrate  --- NCCL's collective-matching contract:
//       if every member of a process group posts a matching collective
//       (same op, same group, same position in the call order), the
//       collective completes. Encoded as the external_body axiom
//       `axiom_matched_implies_no_deadlock`. We do NOT prove this, the
//       same way we do not prove the Triton compiler or the GPU.
//
//   (2) PROVED implication --- if the schedule is DATA-INDEPENDENT
//       (every rank issues the same ordered sequence of (op, group)
//       collectives, with a rank that has nothing to send participating
//       via a zero-sized split rather than skipping the call), then the
//       matching precondition of (1) holds. This is the real lemma
//       `lemma_data_independent_implies_matched` below, and it is what
//       Jun audits.
//
//   (3) INSPECTION leaf    --- the actual Python/Triton schedule really
//       is data-independent: no collective is guarded by a branch on
//       routed data. This is manifest in the straight-line code and is
//       the analogue of the empirical kernel<->DSL correspondence for
//       the fused kernel; it is not a Verus obligation.
//
// So Verus checks (2); (1) and (3) are the declared trust boundary.
//
// Run with:
//   verus --crate-type=lib verus/deadlock_free.rs

use vstd::prelude::*;

verus! {

pub type Rank = nat;

pub enum CollOp {
    AllReduce,
    AllToAll,
    AllGather,
}

/// One collective call: an operation on a process group. Message *sizes*
/// are deliberately omitted --- they are data-dependent but made symmetric
/// by the preceding count-negotiation, and size does not affect whether a
/// collective is matched in op/group/order.
pub struct Step {
    pub op: CollOp,
    pub group: Set<Rank>,
}

/// The ordered sequence of collective calls a given rank issues in one
/// forward pass. Uninterpreted: its concrete value is the schedule the
/// Python/Triton code emits.
pub uninterp spec fn rank_schedule(r: Rank) -> Seq<Step>;

// =====================================================================
// (2a) Data-independence: every rank in the world issues the same
// ordered (op, group) sequence. This is the Verus encoding of "no
// collective is guarded by a data-dependent branch": the call order and
// the group each call names are fixed functions of the program, not of
// any rank's routed tokens.
// =====================================================================

pub open spec fn data_independent(world: Set<Rank>) -> bool {
    forall|r1: Rank, r2: Rank, i: int|
        #![trigger rank_schedule(r1)[i], rank_schedule(r2)[i]]
        (world.contains(r1) && world.contains(r2)
         && 0 <= i < rank_schedule(r1).len()
         && 0 <= i < rank_schedule(r2).len())
        ==> rank_schedule(r1)[i] == rank_schedule(r2)[i]
}

// =====================================================================
// (2b) Matched: the precondition NCCL's contract requires. For every
// call index and every pair of ranks that both participate at that
// index (both belong to the invoked group), they post the same op on
// the same group.
// =====================================================================

pub open spec fn matched(world: Set<Rank>) -> bool {
    forall|r1: Rank, r2: Rank, i: int|
        #![trigger rank_schedule(r1)[i], rank_schedule(r2)[i]]
        (world.contains(r1) && world.contains(r2)
         && 0 <= i < rank_schedule(r1).len()
         && 0 <= i < rank_schedule(r2).len()
         && rank_schedule(r1)[i].group.contains(r1)
         && rank_schedule(r2)[i].group.contains(r2))
        ==> rank_schedule(r1)[i].op == rank_schedule(r2)[i].op
            && rank_schedule(r1)[i].group == rank_schedule(r2)[i].group
}

// =====================================================================
// (2) THE PROVED IMPLICATION: data-independence => matched.
//
// This is the lemma Jun audits. It is short because data-independence
// gives per-index equality of the *entire* Step (op and group) for all
// pairs, and `matched` asks for that same equality on the subset of
// pairs that both participate.
// =====================================================================

pub proof fn lemma_data_independent_implies_matched(world: Set<Rank>)
    requires data_independent(world),
    ensures matched(world),
{
    assert forall|r1: Rank, r2: Rank, i: int|
        #![trigger rank_schedule(r1)[i], rank_schedule(r2)[i]]
        (world.contains(r1) && world.contains(r2)
         && 0 <= i < rank_schedule(r1).len()
         && 0 <= i < rank_schedule(r2).len()
         && rank_schedule(r1)[i].group.contains(r1)
         && rank_schedule(r2)[i].group.contains(r2))
        implies rank_schedule(r1)[i].op == rank_schedule(r2)[i].op
            && rank_schedule(r1)[i].group == rank_schedule(r2)[i].group
    by {
        // data_independent gives Step-equality at index i for r1, r2;
        // equal Steps have equal op and equal group.
        assert(rank_schedule(r1)[i] == rank_schedule(r2)[i]);
    }
}

// =====================================================================
// (1) TRUSTED SUBSTRATE: NCCL's collective-matching contract. A matched
// schedule completes without deadlock. external_body == trusted, not
// proved (like the Triton->hardware lowering).
// =====================================================================

pub uninterp spec fn deadlock_free(world: Set<Rank>) -> bool;

#[verifier::external_body]
pub proof fn axiom_matched_implies_no_deadlock(world: Set<Rank>)
    requires matched(world),
    ensures deadlock_free(world),
{
}

// =====================================================================
// THEOREM: a data-independent collective schedule is deadlock-free.
// Composes the proved implication (2) with the trusted substrate (1).
// The premise `data_independent` for the real schedule is the inspection
// leaf (3).
// =====================================================================

pub proof fn theorem_deadlock_free(world: Set<Rank>)
    requires data_independent(world),
    ensures deadlock_free(world),
{
    lemma_data_independent_implies_matched(world);
    axiom_matched_implies_no_deadlock(world);
}

} // verus!
