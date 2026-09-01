# Ex06_ep_pure — Formal properties for pure expert-parallel MoE

This directory formalizes the correctness of
[`solution.py`](../solution.py) — `EPSparseMoE` in the "pure EP" setting
where each rank owns a **distinct** 1/ep_size slice of tokens (as if a DP
layer preceded the MoE). Under this precondition, dispatch is
**structurally necessary**: a token on rank `r` cannot be processed by an
expert on rank `r'` unless `r` sends it via `all_to_all_variable`.

This is the first exercise where **distributed collectives** are the
correctness content. We prove the routing invariants (from Ex05) plus a
new class of invariants specific to `all_to_all_variable`.

**Style follows [ex05/verification/PROPERTIES.md](../../ex05_moe_baseline/verification/PROPERTIES.md).**

## Abstraction model additions

Ex05's routing model is inherited unchanged. New elements for distributed
comm:

| Concept | Model |
|---|---|
| `Rank` | `nat` in `[0, ep_size)`. |
| `send_sizes(rank)` | `Seq<nat>` of length `ep_size` — how many records rank `r` sends to each other rank in dispatch. |
| `recv_sizes(rank)` | `Seq<nat>` of length `ep_size` — how many records rank `r` receives from each other rank in dispatch. |
| `expert_owner(e)` | `Rank` — which rank owns expert `e`. Bijection modulo partitioning. |
| `all_to_all_symmetric(send_matrix)` | Structural property: for all pairs (i, j), `send_sizes(i)[j] == recv_sizes(j)[i]`. |

`all_to_all_variable` is uninterpreted — we model it as a structural
predicate on `(send_sizes, recv_sizes)` pairs, plus an axiom about its
data-content post-condition.

## Divisibility invariants (asserted at construction)

- `num_experts % ep_size == 0` — clean expert-sharding partition.
- `experts_per_rank = num_experts / ep_size`.
- `expert_start(r) = r * experts_per_rank`; `expert_end(r) = (r+1) * experts_per_rank`.

## Properties to verify

### EP1 — Expert sharding disjointness

$$
\forall r \ne r' : [\text{expert\_start}(r), \text{expert\_end}(r)) \cap [\text{expert\_start}(r'), \text{expert\_end}(r')) = \emptyset
$$

**Proof**: interval-disjointness by monotonicity, mirroring Ex01's C2.

### EP2 — Expert coverage

$$
\bigcup_{r = 0}^{\text{ep\_size} - 1} [\text{expert\_start}(r), \text{expert\_end}(r)) = [0, \text{num\_experts})
$$

**Proof**: by `experts_per_rank * ep_size == num_experts`.

### EP3 — Dispatch symmetry (all_to_all invariant)

For every pair of ranks (i, j), the count of records rank i sends to rank j
in dispatch equals the count rank j receives from rank i:

$$
\forall i, j : \text{send\_sizes}(i)[j] = \text{recv\_sizes}(j)[i]
$$

**Proof**: definitional postcondition of the negotiation collective
(`all_to_all_single` on the count arrays that runs BEFORE the data
`all_to_all_variable`). We axiomatize this as `axiom_all_to_all_count_symmetric`.

### EP4 — Token conservation across dispatch

Total records sent equals total records received in the dispatch step,
globally:

$$
\sum_{i,j} \text{send\_sizes}(i)[j] = \sum_{i,j} \text{recv\_sizes}(i)[j]
$$

**Proof**: consequence of EP3 (summing over the pair-wise equality).

### EP5 — Per-rank dispatch-and-combine round-trip

Each rank r sends `sum(send_sizes(r))` records in dispatch and receives
`sum(send_sizes(r))` records in combine (the reverse dispatch). Combined,
the round-trip preserves every original `(token, expert)` pair.

**Proof**: EP3 + EP4 + the reverse-dispatch swap that `combine` performs.

### EP6 — Deadlock-freedom

For every collective call in the schedule, all ranks in the group agree on
the sequence (order and shape) of collective operations. Formally: the
schedule is a straight-line sequence of `all_to_all_single` /
`all_to_all_variable` / `all_reduce` calls, identical on every rank up to
the argument-shape lists.

**Proof**: the Python schedule is a straight-line function of the input
shape (no data-dependent branches on any rank's private state), so all
ranks reach the same collective call site with matching shapes. Formalized
as a structural property of the abstract program.

### EP7 — Routing correctness (inherited from Ex05)

The output of `EPSparseMoE.forward(local_x)` on rank `r` equals the
per-rank slice of `MoE_spec(x_full, W, g, k)`, up to `approx_eq` tolerance
— provided the dispatch-then-compute-then-combine schedule preserves
token conservation (EP4) and per-rank expert-set disjointness (EP1, EP2).

**Proof**: composition of EP1-EP5 with Ex05's E1 (per-expert local
compute correctness) and RT1-RT3 (routing invariants).

Stated as a stub — the full mechanization requires modeling the abstract
data-content post-condition of `all_to_all_variable`, which is a
substantial addition to the axiom base.

## What each tool proves — this exercise

Same three-tool pattern as Ex01-05. Verus proof shipped first.

## Correspondence to Python

1. `EPSparseMoE.__init__` sets `expert_start = experts_per_rank * ep_rank`
   and `expert_end = expert_start + experts_per_rank`, which matches EP1
   + EP2 by direct arithmetic.
2. `EPSparseMoE.forward` calls
   `dist.all_to_all_single(recv_cnts_buf, send_cnts_buf, group=self.ep_group)`
   BEFORE the data dispatch — this establishes EP3 at runtime.
3. Deadlock-freedom (EP6) is a straight-line-schedule property of the
   Python code; visible by inspection.
