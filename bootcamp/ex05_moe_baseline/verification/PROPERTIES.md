# Ex05 — Formal properties for single-GPU MoE (naive + permuted variants)

This directory formalizes the correctness of both variants of Ex05:

- [`solution_a.py`](../solution_a.py) — `NaiveSparseMoE`, per-expert loop
  with scatter/gather at every step.
- [`solution_b.py`](../solution_b.py) — `PermutedSparseMoE`, permuted +
  grouped compute (the algorithmic bridge to Ex06/Ex09).

Both compute the same math; they differ in *data layout*. The verification
target is: **both compute the abstract MoE function, up to floating-point
accumulation order**.

**Style follows [ex01/verification/PROPERTIES.md](../../ex01_linear_tp/verification/PROPERTIES.md).**
This is the first exercise where linear-algebra properties are NOT the
verification content. What we prove here is **routing correctness** —
token conservation, top-k weight normalization, permutation invertibility
— which is the load-bearing invariant for every downstream MoE proof
(Ex06, Ex07, Ex09).

## Abstraction model additions

This is the first exercise where the tensor content of the input isn't
central. We introduce:

| Concept | Model |
|---|---|
| `top_k_ids` | `Seq<Seq<ExpertId>>` — for each token, `top_k` selected experts. Shape `[N, top_k]`. |
| `top_k_weights` | `Seq<Seq<Weight>>` — normalized routing weights, shape `[N, top_k]`. |
| `Weight` | Abstracted as `int` (like `Element` in Ex01) — we're proving conservation, not numerics. |
| `sorted_token_ids` | `Seq<TokenId>` — the argsort permutation over the flattened `(token, expert)` pairs. Length `N * top_k`. |
| `sorted_expert_ids` | `Seq<ExpertId>` — the corresponding sorted expert ids. Length `N * top_k`. |
| `offsets` | `Seq<nat>` — CSR-style expert boundaries, length `num_experts + 1`, monotone non-decreasing, `offsets[0] == 0`, `offsets[num_experts] == N * top_k`. |
| `expert_apply(e, x_slice)` | Uninterpreted spec function — one expert's MLP applied to a batch of tokens. |

## Abstract MoE spec function

The single semantic reference every MoE variant refines:

$$
\mathrm{MoE\_spec}(x, W, g, k)[i] = \sum_{j=0}^{k-1} w_{ij} \cdot \mathrm{expert\_apply}(e_{ij}, x[i])
$$

where $(e_{ij}, w_{ij}) := \mathrm{topk\_softmax}(g \cdot x[i], k)$ and
$\sum_j w_{ij} = 1$ (when `norm_topk_prob = True`).

Both `NaiveSparseMoE` and `PermutedSparseMoE` refine this spec up to
floating-point reduction-order tolerance. The tolerance is an axiomatic
`approx_eq` predicate; the schedule-level equivalence is the load-bearing
proof content.

## Properties to verify — routing invariants

### RT1 — Token conservation

The flattened `(token, expert)` pairs partition into `N * top_k` triples:

$$
\sum_{e = 0}^{\text{num\_experts} - 1} (\text{offsets}[e+1] - \text{offsets}[e]) = N \cdot \text{top\_k}
$$

Equivalently: `offsets[num_experts] - offsets[0] == N * top_k`. **Proof**:
by the definition of `bincount(sorted_expert_ids, minlength=num_experts)`
and `cumsum`, the final offset equals the sum of counts, which equals the
length of `sorted_expert_ids`, which equals `N * top_k`.

### RT2 — Offset monotonicity

$$
\forall e \in [0, \text{num\_experts}) : \text{offsets}[e] \le \text{offsets}[e+1]
$$

**Proof**: `cumsum` of a non-negative sequence is monotone non-decreasing.

### RT3 — Top-k weight normalization (when `norm_topk_prob`)

$$
\forall i \in [0, N) : \sum_{j = 0}^{\text{top\_k} - 1} \text{top\_k\_weights}[i][j] = 1
$$

**Proof**: `softmax(top_k_raw_logits)` sums to 1 by definition of softmax
(exp / sum-of-exps). We axiomatize this as an axiom on `softmax`.

### RT4 — Permutation invertibility

The argsort permutation `sort_perm` is a bijection over `[0, N * top_k)`:

$$
\forall p, p' \in [0, N \cdot \text{top\_k}) : p \ne p' \implies \text{sort\_perm}[p] \ne \text{sort\_perm}[p']
$$

and

$$
\{\text{sort\_perm}[p] : p \in [0, N \cdot \text{top\_k})\} = [0, N \cdot \text{top\_k})
$$

**Proof**: any sort produces a permutation; the abstract model captures
this by axiomatizing `argsort` to return a permutation. Formally, we prove
via `Seq::to_multiset` equality.

## Properties to verify — equivalence to abstract spec

### E1 — `NaiveSparseMoE` refines `MoE_spec`

For each token index `i`, the naive implementation's output at position
`i` equals `MoE_spec(x, W, g, k)[i]` up to declared `approx_eq` tolerance.

**Proof composition**:
1. RT3 gives that `top_k_weights` sums to 1 per token.
2. The naive loop over experts computes
   `output[i] = sum over e: (i in expert_e's token_ids) * w[i, e] * expert_apply(e, x[i])`.
3. This directly matches `MoE_spec[i]` by definition of top-k routing.

Stated with `approx_eq` because summation order in the loop differs from
the reference oracle's ordering.

### E2 — `PermutedSparseMoE` refines `MoE_spec` (approx_eq to `NaiveSparseMoE`)

For each token index `i`, the permuted implementation's output at position
`i` equals the naive implementation's output at position `i` up to
`approx_eq` tolerance.

**Proof composition**:
1. RT1 + RT2 + RT4 give that the permutation collects all `(token, expert)`
   pairs into per-expert contiguous slices without loss.
2. The grouped compute step applies each expert to its slice, then
   scatter-adds weighted outputs at the original token positions.
3. Modulo permutation-induced reordering of the summation, this equals
   the naive per-expert loop. The reordering is exactly what `approx_eq`
   absorbs.

## What each tool proves — this exercise

Same three-tool pattern as Ex01-04. Verus proof shipped first.

## Correspondence to Python

1. `NaiveSparseMoE.forward` in Python computes exactly the per-token
   per-expert loop that E1 asserts equals `MoE_spec`.
2. `PermutedSparseMoE.forward` in Python computes the permutation +
   grouped + scatter-add pattern that E2 asserts equals `NaiveSparseMoE`.
3. Both use PyTorch's `torch.topk` and `torch.sort(stable=True)`, whose
   contracts match our axiomatic `argsort` and `topk` spec functions.
