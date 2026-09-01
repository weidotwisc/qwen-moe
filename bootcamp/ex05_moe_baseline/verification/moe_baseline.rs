// moe_baseline.rs
//
// Verus attempt at single-GPU MoE correctness properties for Ex05.
// Proves:
//   RT1 (token conservation via offsets)
//   RT2 (offset monotonicity)
//   RT3 (top-k weight normalization via axiom)
//   E1 stubbed (NaiveSparseMoE refines MoE_spec)
//   E2 stubbed (PermutedSparseMoE refines NaiveSparseMoE up to approx_eq)
//
// Focus: routing correctness (token conservation, monotonic offsets, top-k
// normalization). These are the load-bearing invariants for every downstream
// MoE proof (Ex06, Ex07, Ex09).
//
// Run with:
//   verus moe_baseline.rs

use vstd::prelude::*;

verus! {

// =====================================================================
// §1 — Types.
// =====================================================================

pub type TokenId = nat;
pub type ExpertId = nat;
pub type Weight = int;   // abstract; not reasoning about numerics

// =====================================================================
// §2 — Bincount and cumsum specs (used to construct offsets).
// =====================================================================

/// bincount(sorted_expert_ids, minlength=num_experts): for each expert e in
/// [0, num_experts), returns the count of occurrences of e in sorted_expert_ids.
pub open spec fn count_at(seq: Seq<ExpertId>, e: ExpertId) -> nat
    decreases seq.len()
{
    if seq.len() == 0 {
        0nat
    } else {
        (if seq[0] == e { 1nat } else { 0nat })
            + count_at(seq.subrange(1, seq.len() as int), e)
    }
}

pub open spec fn bincount_row(seq: Seq<ExpertId>, num_experts: nat, idx: nat) -> nat
    recommends idx < num_experts,
{
    count_at(seq, idx as ExpertId)
}

/// Recursive cumsum: prefix sum of a Seq<nat>, resulting in a Seq of length n+1.
pub open spec fn cumsum(counts: Seq<nat>) -> Seq<nat>
    decreases counts.len(),
{
    if counts.len() == 0 {
        seq![0nat]
    } else {
        let init = cumsum(counts.subrange(0, counts.len() - 1));
        init.push(init.last() + counts[counts.len() - 1])
    }
}

pub proof fn lemma_cumsum_len(counts: Seq<nat>)
    ensures cumsum(counts).len() == counts.len() + 1,
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        lemma_cumsum_len(counts.subrange(0, counts.len() - 1));
    }
}

pub proof fn lemma_cumsum_first_is_zero(counts: Seq<nat>)
    ensures cumsum(counts)[0] == 0nat,
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        lemma_cumsum_first_is_zero(counts.subrange(0, counts.len() - 1));
    }
}

pub proof fn lemma_cumsum_monotone(counts: Seq<nat>)
    ensures forall|i: int| 0 <= i < counts.len() as int
                ==> #[trigger] cumsum(counts)[i] <= cumsum(counts)[i + 1],
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        let init = counts.subrange(0, counts.len() - 1);
        lemma_cumsum_monotone(init);
        lemma_cumsum_len(init);
        // cumsum(counts) = cumsum(init).push(cumsum(init).last() + counts.last())
        let c = cumsum(counts);
        let ci = cumsum(init);
        assert(c =~= ci.push(ci.last() + counts[counts.len() - 1]));
        assert forall|i: int| 0 <= i < counts.len() as int
            implies #[trigger] c[i] <= c[i + 1] by {
            if i < (counts.len() - 1) as int {
                // c[i] == ci[i], c[i+1] == ci[i+1], and ci is monotone by IH.
                assert(c[i] == ci[i]);
                assert(c[i + 1] == ci[i + 1]);
            } else {
                // i == counts.len() - 1, so c[i] == ci.last(),
                // c[i+1] == ci.last() + counts[len-1] >= ci.last().
                assert(c[i] == ci.last());
                assert(c[i + 1] == ci.last() + counts[counts.len() - 1]);
            }
        }
    }
}

// =====================================================================
// §3 — Property RT2: offset monotonicity.
// =====================================================================

pub proof fn rt2_offset_monotonicity(counts: Seq<nat>)
    ensures
        forall|e: int| 0 <= e < counts.len() as int
            ==> #[trigger] cumsum(counts)[e] <= cumsum(counts)[e + 1],
{
    lemma_cumsum_monotone(counts);
}

// =====================================================================
// §4 — Property RT1: token conservation.
//
// The final offset equals the total count, which equals N * top_k.
// We prove the cumsum-total-equals-sum fact; the connection to
// N * top_k comes from bincount's spec (sum of counts == input length).
// =====================================================================

pub open spec fn seq_sum(s: Seq<nat>) -> nat
    decreases s.len(),
{
    if s.len() == 0 {
        0nat
    } else {
        s[0] + seq_sum(s.subrange(1, s.len() as int))
    }
}

pub proof fn lemma_cumsum_last_is_sum(counts: Seq<nat>)
    ensures cumsum(counts).last() == seq_sum(counts),
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        let init = counts.subrange(0, counts.len() - 1);
        lemma_cumsum_last_is_sum(init);
        lemma_seq_sum_split_last(counts);
    }
}

pub proof fn lemma_seq_sum_split_last(s: Seq<nat>)
    requires s.len() > 0,
    ensures seq_sum(s) == seq_sum(s.subrange(0, s.len() - 1)) + s[s.len() - 1],
    decreases s.len(),
{
    let n = s.len() as int;
    if s.len() == 1 {
        assert(s.subrange(0, 0) =~= Seq::<nat>::empty());
        assert(seq_sum(Seq::<nat>::empty()) == 0);
        assert(seq_sum(s) == s[0] + seq_sum(s.subrange(1, 1)));
        assert(s.subrange(1, 1) =~= Seq::<nat>::empty());
    } else {
        // Recurse on s.subrange(1, n) (tail).
        let tail = s.subrange(1, n);
        lemma_seq_sum_split_last(tail);
        // seq_sum(s) = s[0] + seq_sum(tail)
        //            = s[0] + seq_sum(tail.subrange(0, tail.len()-1)) + tail.last()
        // We need seq_sum(s) = seq_sum(s.subrange(0, n-1)) + s.last().
        // s.subrange(0, n-1) = s[0] :: tail.subrange(0, tail.len()-1) (structurally).
        assert(tail.len() == (n - 1) as nat);
        assert(tail.last() == s[n - 1]);
        // Establish the subrange equality via extensional equality.
        let init_of_s = s.subrange(0, n - 1);
        let init_of_tail = tail.subrange(0, tail.len() - 1);
        assert(init_of_s.len() == init_of_tail.len() + 1);
        assert(init_of_s[0] == s[0]);
        assert forall|i: int| 1 <= i < init_of_s.len()
            implies init_of_s[i] == init_of_tail[i - 1] by {
            assert(init_of_s[i] == s[i]);
            assert(init_of_tail[i - 1] == tail[i - 1]);
            assert(tail[i - 1] == s[i]);
        }
        // seq_sum(init_of_s) = s[0] + seq_sum(tail_of_init_of_s = init_of_tail).
        lemma_seq_sum_prepend_equal(init_of_s, init_of_tail);
    }
}

pub proof fn lemma_seq_sum_prepend_equal(a: Seq<nat>, b: Seq<nat>)
    requires
        a.len() == b.len() + 1,
        forall|i: int| 1 <= i < a.len() ==> a[i] == b[i - 1],
    ensures
        seq_sum(a) == a[0] + seq_sum(b),
{
    let tail = a.subrange(1, a.len() as int);
    assert(tail =~= b);
}

/// RT1: the final offset equals `total_tokens` (= N * top_k).
///
/// Under the abstract model: `offsets = cumsum(bincount(sorted_expert_ids, num_experts))`.
/// The sum of the bincount values equals the length of `sorted_expert_ids`, which is
/// N * top_k. Hence `offsets.last() == N * top_k`.
pub proof fn rt1_token_conservation(counts: Seq<nat>, total_tokens: nat)
    requires seq_sum(counts) == total_tokens,
    ensures cumsum(counts).last() == total_tokens,
{
    lemma_cumsum_last_is_sum(counts);
}

// =====================================================================
// §5 — Property RT3: top-k weight normalization (via axiom on softmax).
// =====================================================================

/// The top-k weights per token, uninterpreted (comes from softmax of top-k logits).
pub uninterp spec fn top_k_weights(top_k_logits: Seq<Weight>) -> Seq<Weight>;

/// Axiom SOFTMAX_SUM: softmax of any input sums to 1.
/// We represent "1" as the integer 1 since Weight is int (abstracted).
#[verifier::external_body]
pub proof fn axiom_softmax_sums_to_one(logits: Seq<Weight>)
    ensures seq_sum_of_weights(top_k_weights(logits)) == 1int,
{}

/// Sum of a Seq<Weight> as int (for the axiom).
pub open spec fn seq_sum_of_weights(s: Seq<Weight>) -> int
    decreases s.len(),
{
    if s.len() == 0 {
        0int
    } else {
        s[0] + seq_sum_of_weights(s.subrange(1, s.len() as int))
    }
}

pub proof fn rt3_topk_weight_normalization(logits: Seq<Weight>)
    ensures seq_sum_of_weights(top_k_weights(logits)) == 1int,
{
    axiom_softmax_sums_to_one(logits);
}

// =====================================================================
// §6 — MoE_spec: the abstract semantic reference.
// =====================================================================

pub uninterp spec fn expert_apply(e: ExpertId, x_token: Seq<Weight>) -> Seq<Weight>;

pub uninterp spec fn moe_spec_output(
    x: Seq<Seq<Weight>>,
    ids: Seq<Seq<ExpertId>>,
    weights: Seq<Seq<Weight>>,
    token_i: nat,
) -> Seq<Weight>;

// =====================================================================
// §7 — Property E1: NaiveSparseMoE refines MoE_spec (external stub).
//
// Proof composition: the per-expert loop in `NaiveSparseMoE.forward`
// computes, for each token i, the sum over j in [0, top_k) of
// w[i, j] * expert_apply(ids[i][j], x[i]). This IS MoE_spec[i] by
// definition. The `approx_eq` tolerance absorbs summation-order differences.
// Left as external because it's the whole-forward equivalence, which
// requires a lot of scaffolding (loop invariant + tolerance composition).
// =====================================================================

#[verifier::external_body]
pub proof fn e1_naive_refines_spec_stub()
    ensures true,
{}

// =====================================================================
// §8 — Property E2: PermutedSparseMoE refines NaiveSparseMoE (external stub).
//
// Requires proving that the permutation-then-group-then-scatter is
// extensionally equal (up to reordering of a commutative sum) to the
// naive per-token per-expert loop. RT1+RT2 give the necessary structural
// properties of `offsets` and `sorted_*` sequences; RT4 (permutation
// bijection) gives that no data is dropped. The final numerical step
// (weight-and-scatter) is a commutative sum; equivalence is up to
// `approx_eq` tolerance.
// =====================================================================

#[verifier::external_body]
pub proof fn e2_permuted_refines_naive_stub()
    ensures true,
{}

} // verus!

fn main() {
    println!("Verified single-GPU MoE routing properties (Verus).");
}
