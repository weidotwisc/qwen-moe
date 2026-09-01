# Ex06_ep — Formal properties for the lean (all_reduce) EP variant

This directory formalizes the correctness of
[`reference_lean.py`](../reference_lean.py) — `EPSparseMoE` in the
**replicated-input** setting where every rank sees the full `[N, H]`
tensor. Under this precondition, dispatch is redundant: each rank can
filter routing records to its LOCAL experts, run local compute, and use
**one `all_reduce`** to sum partial contributions.

**This variant is the paper's centerpiece efficiency claim.** It's the
schedule the paper's composition theorem certifies as functionally
equivalent to the dispatch-based variant (Ex06_ep_pure), with measured
speedups of 1.09-1.26× on-node NVLink and 1.72× cross-node TCP over
dispatch.

**Style follows [ex06_ep_pure/verification/PROPERTIES.md](../../ex06_ep_pure/verification/PROPERTIES.md).**
Same routing-invariant + expert-partition scaffolding; different
collective structure (one all_reduce vs two all_to_alls).

## Attribution note

`reference_lean.py` was AI-drafted by Claude in an earlier session (not
manually written by Wei). It is verified here for the same reason the
manually-authored microbenchmarks are verified: **the paper's contribution
is the AI-drafted / human-audited workflow applied to a real brown-field
task, and the lean variant is a critical part of the composition-theorem
target set.** The audit protocol is identical — Jun reads the Verus
contracts and the Python reference oracle side-by-side, confirms the
correspondence, and signs off.

## Abstraction model

Ex05's routing model + Ex06_ep_pure's expert-partition model, plus a new
element:

| Concept | Model |
|---|---|
| `Replicated(x, ep_group)` | Precondition: `x` is identical on every rank in ep_group. |
| `local_mask(rank, top_k_experts, expert_start(rank), expert_end(rank))` | Which (token, expert) pairs on this rank belong to its local experts. |
| `partial_output(rank)` | The `[N, H]` tensor rank r builds by scatter-adding weighted expert outputs at their token positions. Zero at token positions rank r doesn't contribute to. |
| `all_reduce_sum` | Uninterpreted; models `dist.all_reduce(op=SUM)`. Postcondition: output on every rank is `sum over r' in group of partial_output(r')`. |

## Divisibility invariants (asserted at construction)

- `num_experts % ep_size == 0`.
- `experts_per_rank = num_experts / ep_size`.
- `expert_start(r) = r * experts_per_rank`.
- `expert_end(r) = (r+1) * experts_per_rank`.

## Properties to verify

### L1 — Precondition: input is Replicated across ep_group

The lean variant's `forward` **requires** `x` to be Replicated. If this
precondition is violated, the schedule is unsound: different ranks would
compute different partial outputs from different inputs, and the
all_reduce sum would be meaningless.

**Formal statement**: `forall r, r' in ep_group : x_on(r) == x_on(r')`.

**Proof**: precondition. Downstream compositions (Ex07 hybrid) discharge
it by chaining Ex04's R4 (TP-RowParallelLinear all-reduce establishes
replication).

### L2 — Local-mask disjointness

For every `(token, expert)` pair `(i, e)` produced by the router, exactly
one rank considers it "local":

$$
\forall (i, e) : \left| \{ r \in \text{ep\_group} : \text{expert\_start}(r) \le e < \text{expert\_end}(r) \} \right| = 1
$$

**Proof**: by Ex06_ep_pure's EP1 (expert-sharding disjointness) +
EP2 (expert coverage) applied to the routing records. Each expert `e`
is owned by exactly one rank, so exactly that rank's `local_mask` sets
`e` to True.

### L3 — Partial output at non-routed positions is zero

For any rank `r` and any token index `i` that has no `(i, e)` pair with
`e` in rank `r`'s expert range:

$$
\text{partial\_output}(r)[i] = 0
$$

**Proof**: `partial_output = torch.zeros(...)` initially; `index_add_` at
`sorted_token_ids` only writes to indices that appear in
`sorted_token_ids`, which are precisely the tokens routed to rank r's
local experts. Positions NOT in `sorted_token_ids` remain zero.

### L4 — Sum of partial outputs equals full MoE output

$$
\sum_{r = 0}^{\text{ep\_size} - 1} \text{partial\_output}(r)[i]
= \sum_{(e_{ij}, w_{ij}) \in \text{top\_k}(x[i])} w_{ij} \cdot \text{expert\_apply}(e_{ij}, x[i])
$$

for every token index `i`. RHS is the definition of `MoE_spec`.

**Proof composition**:
1. By L2, each `(i, e)` pair contributes to exactly one rank's
   partial_output.
2. By construction of `partial_output`, rank `r` adds `w[i,e] *
   expert_apply(e, x[i])` to `partial_output(r)[i]` for each
   `(i, e)` with `e` in rank `r`'s expert range.
3. Summing over ranks recovers the full sum in the RHS.

### L5 — Post-condition: output is Replicated

After `dist.all_reduce(partial_output, SUM, group=ep_group)`:

$$
\forall r, r' \in \text{ep\_group} : \text{output\_on}(r) = \text{output\_on}(r')
$$

**Proof**: axiom `axiom_all_reduce_sum_replicated` (from the trusted
axiom base).

### L6 — Refinement to MoE_spec

For every token index `i` on any rank:

$$
\text{output}[i] \approx_{\text{atol,rtol}} \text{MoE\_spec}(x, W, g, k)[i]
$$

**Proof composition**: L4 (sum equals MoE_spec RHS pointwise) + L5
(all_reduce sum is replicated on every rank). Up to `approx_eq`
tolerance because the summation order in `all_reduce` may differ from
the reference oracle's natural order.

Stated as a stub — full mechanization requires the composition machinery
also used by Ex06_ep_pure's EP7.

## What each tool proves — this exercise

Same three-tool pattern as Ex01-07/Ex09. Verus proof shipped first.

## Correspondence to Python

1. `reference_lean.py::EPSparseMoE.__init__` sets
   `expert_start = ep_rank * experts_per_rank` and
   `expert_end = expert_start + experts_per_rank`. Matches L2's
   partition directly.
2. `reference_lean.py::EPSparseMoE.forward` Phase 2 constructs
   `local_mask = (top_k_experts_flat >= expert_start) & (top_k_experts_flat < expert_end)`,
   which is a bit-mask indexing into `[0, ep_size * experts_per_rank)`
   that satisfies L2 by construction.
3. `partial_output = torch.zeros(...)` then `index_add_(0, sorted_token_ids, expert_out)`
   satisfies L3 (zeros at non-indexed positions).
4. `dist.all_reduce(partial_output, op=dist.ReduceOp.SUM, group=self.group)`
   is the collective whose postcondition L5 axiomatizes.

## The paper's composition-theorem claim (sketch)

Combining Ex06_ep_pure's EP7 (dispatch-based routing correctness) with
this file's L6 (lean-based routing correctness), the paper's composition
theorem states:

> **Under the precondition `Replicated(x, ep_group)`, the lean variant
> and the dispatch variant produce outputs that are `approx_eq` to each
> other. Both refine the same `MoE_spec` up to tolerance.**

This is the equivalence claim the paper's Contribution 2 rests on. Its
full mechanization composes L1-L6 (this file) with EP1-EP7 (Ex06_ep_pure).

## Correctness of the abstraction

Two questions:

1. **Is the axiom about `all_reduce_sum` sound?**
   Yes — `dist.all_reduce(op=SUM)` is defined by PyTorch's distributed
   contract as: every rank in `group` receives the elementwise sum of
   all ranks' input tensors. This is standard collective semantics.

2. **Is the local_mask filter faithful?**
   Yes — Python's `(top_k_experts_flat >= expert_start) &
   (top_k_experts_flat < expert_end)` is an elementwise integer
   comparison producing a bool tensor. Every routing record `(i, e)`
   with `e in [expert_start, expert_end)` is marked True, and none
   others. This matches L2 exactly.
