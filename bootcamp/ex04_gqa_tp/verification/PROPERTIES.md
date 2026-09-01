# Ex04 — Formal properties for GQA under TP with KV-head replication

This directory formalizes the correctness of
[`solution.py`](../solution.py) — `QKVParallelLinearGQA` + `TPGQA`, which
extends Ex03's MHA to grouped-query attention with a KV-replication rule
that unblocks nanovllm-jun for TP-8 on Qwen3-30B-A3B (where
`num_kv_heads = 4 < tp_size = 8`).

**Style follows [ex03/verification/PROPERTIES.md](../../ex03_mha_tp/verification/PROPERTIES.md).**
The Q-projection side is structurally identical to Ex03 (properties Q1-Q4).
The novel content is the **KV-replication invariant** and the semantics of
`repeat_interleave` on the head axis.

## Abstraction model additions

Ex03's abstractions are inherited unchanged. Two new elements:

| Concept | Model |
|---|---|
| `num_kv_replicas` | `max(1, tp_size / num_kv_heads)` — how many ranks share one KV head when `num_kv_heads < tp_size`. |
| `kv_slot(r) := r / num_kv_replicas` | Which KV-head partition rank `r` reads from. Multiple ranks may share the same slot. |
| `kv_shard(w_kv, r)` | The KV slice this rank stores. Equal to `shard(w_kv, kv_slot(r), num_unique_kv_slots)` where `num_unique_kv_slots = num_kv_heads / num_kv_heads_per_rank`. |
| `repeat_interleave(t, n_rep)` | Uninterpreted spec function; models the per-head duplication broadcasting K/V up to Q's head count. |

`repeat_interleave` is uninterpreted because its semantics — "each of the
first-axis entries is repeated `n_rep` times in place" — is a structural
operation, not part of the schedule correctness content. Its key
axiomatic property (**axiom RI1**, below) is all the proof needs.

**Axiom RI1 — repeat_interleave preserves per-original-index content**:

$$
\forall t, n : \quad \mathrm{repeat\_interleave}(t, n)[i \cdot n + k] = t[i]
\quad \text{for all } 0 \le i < |t|, \; 0 \le k < n
$$

Equivalently: the concatenated form
$\mathrm{repeat\_interleave}(t, n) = t[0]^n \parallel t[1]^n \parallel \cdots$
where $t[i]^n$ denotes $t[i]$ repeated $n$ times.

## Divisibility invariants (asserted at construction)

For `num_heads = H_q`, `num_kv_heads = H_kv`, `tp_size = P`, `head_dim = D`:

- $H_q \bmod P = 0$ — Q shards cleanly.
- $H_{kv} \bmod P = 0 \vee P \bmod H_{kv} = 0$ — either KV shards cleanly (no replication) or `tp_size` is a multiple of `num_kv_heads` (clean replication).

These make `num_kv_heads_per_rank`, `num_kv_replicas`, and `n_rep` all
well-defined naturals.

## Properties to verify — QKVParallelLinearGQA

### G1 — Q-side inherits from Ex03

$$
\text{self.weight}[0 : q\_\mathrm{shard}] = q\_\mathrm{shard\_of\_w}_q(r)
$$

The Q sub-region of `self.weight` on rank `r` equals the standard
Ex03-style shard of `w_q`. **Proof**: definitional; same as Ex03's Q1
restricted to the Q sub-region.

### G2 — KV replication invariant

For any two ranks `r`, `r'` with `kv_slot(r) == kv_slot(r')`:

$$
\text{self.weight}[q\_\mathrm{shard} : q\_\mathrm{shard} + k\_\mathrm{shard}]
\text{ on rank } r
= \text{self.weight}[q\_\mathrm{shard} : q\_\mathrm{shard} + k\_\mathrm{shard}]
\text{ on rank } r'
$$

i.e., replica-siblings hold *bit-identical* K weights (and similarly for V).
This is the load-bearing new invariant that ex04's `weight_loader`
introduces. **Proof**: definitional — `weight_loader("k")` on rank r stores
`w_k.chunk(num_kv_heads, dim=0)[r // num_kv_replicas]`, and two ranks with
the same `r // num_kv_replicas` receive the same chunk.

### G3 — KV shards partition (coarser)

Gathering K shards *by unique kv_slot* (deduplicating replicas) reconstructs
`w_k`:

$$
\bigparallel_{s=0}^{num\_kv\_slots - 1} k\_\mathrm{shard\_of\_slot}(s) = w_k
$$

where `k_shard_of_slot(s) := shard(w_k, s, num_kv_slots)`.
Similarly for V. **Proof**: application of ex01's C1 with `num_kv_slots`
in place of `tp_size`.

### G4 — Three-projection weight_loader post-condition

After `weight_loader(w_q, "q")`, `weight_loader(w_k, "k")`,
`weight_loader(w_v, "v")` on rank r:

$$
\text{self.weight.data} = q\_\mathrm{shard\_of\_w}_q(r) + k\_\mathrm{shard\_of\_slot}(\mathrm{kv\_slot}(r)) + v\_\mathrm{shard\_of\_slot}(\mathrm{kv\_slot}(r))
$$

The three narrow-copy calls populate disjoint regions with the shards
determined by rank (for Q) and by KV slot (for K and V). **Proof**:
definitional composition of G1 and G2.

## Properties to verify — TPGQA composition

Let `q_r`, `k_slot(r)`, `v_slot(r)` be the per-rank / per-slot projection
outputs of the QKV forward. Let `n_rep := num_heads_per_rank /
num_kv_heads_per_rank`. Define:

- $q_r := \mathrm{matmul}(x, \mathrm{transpose}(q\_\mathrm{shard\_of\_w}_q(r)))$
- $k_r := \mathrm{repeat\_interleave}(\mathrm{matmul}(x, \mathrm{transpose}(k\_\mathrm{shard\_of\_slot}(\mathrm{kv\_slot}(r)))), n\_\mathrm{rep})$
- $v_r$ analogous
- $a_r := \mathrm{attention}(q_r, k_r, v_r)$

### R1 — After repeat_interleave, K/V have the same head count as Q

$$
|k_r| = |q_r| \quad \text{and} \quad |v_r| = |q_r|
$$

**Proof**: `|repeat_interleave(t, n)| = n * |t|`. Combined with
`|q_r| = num_heads_per_rank * head_dim` and
`|matmul(x, k_shard_of_slot(...))| = num_kv_heads_per_rank * head_dim`,
the multiplication by `n_rep = num_heads_per_rank / num_kv_heads_per_rank`
gives the required equality.

### R2 — Two replica-siblings compute the same attention output

For ranks r, r' with `kv_slot(r) == kv_slot(r')`, if they receive the same
`q_r == q_{r'}` (they don't — Q is uniquely sharded — but suppose we
mentally rerun with the *other* rank's Q), then their `a_r` values would
be identical *up to which Q heads are attending*, because K and V (after
repeat_interleave) are bit-identical.

Precisely: attention over any two ranks sharing the same KV slot uses
identical (K, V) — the redundancy of KV replication is exactly this
"per-slot K and V are the same across replica-siblings" fact. This is
required for GQA's correctness under KV replication because otherwise
different ranks in the same KV slot would attend against different KV
data.

**Proof**: G2 applied to K and V, plus the fact that `repeat_interleave`
of identical inputs produces identical outputs.

### R3 — Block correctness (via axioms M1, M2, A1, RI1)

The full block output equals the unsharded GQA output.

**Proof composition** (documented, not machine-checked here):
1. G3 + G4 + Q3-style application of axiom M1 gives the merged QKV forward.
2. R1 discharges the head-count-agreement precondition of `attention`.
3. R2 discharges the "K/V per-slot consistency" needed for GQA.
4. Ex03's A1 (attention commutes with head-shard) combined with the
   post-repeat_interleave head count gives attention gather commutativity.
5. Ex01's R4 (RowParallelLinear all-reduce) finishes.

Stated as a stub in the Verus proof; the full mechanization is deferred.

## What each tool proves — this exercise

Same three-tool pattern as Ex01-03. Verus proof shipped first.

## Correspondence to Python

The Verus proof abstracts `kv_slot`, `repeat_interleave`, `attention`, and
matmul. Correspondence:

1. `weight_loader("k")` in Python computes
   `full_weight.chunk(num_kv_heads, dim=0)[tp_rank // kv_replicas]`,
   which matches `k_shard_of_slot(kv_slot(r))` in the Verus model. G2's
   replica invariant is a definitional consequence.
2. `repeat_interleave(k, num_q_heads_per_rank // num_kv_heads_per_rank, dim=1)`
   in Python (line 270 of `solution.py`) corresponds to
   `repeat_interleave(k_r, n_rep)` in the Verus model. Axiom RI1 gives
   the per-original-index content property.
3. Everything else (RoPE folded into attention, SDPA head-locality) is
   identical to Ex03's abstraction.

## Correctness of the abstraction

Two questions:

1. **Is axiom RI1 (repeat_interleave preserves per-original-index content) true?**
   Yes — by definition of PyTorch's `Tensor.repeat_interleave(n, dim=1)`.
   This is not a numerical property; it's a purely structural indexing
   fact that PyTorch's documentation states precisely.

2. **Does the KV-replication implementation faithfully realize G2?**
   Yes — the Python `weight_loader("k")` on rank `r` uses
   `full_weight.chunk(num_kv_heads, dim=0)[r // num_kv_replicas]`. Two
   ranks `r`, `r'` with the same `r // num_kv_replicas` extract the same
   chunk. This is straightforward Python arithmetic that any auditor can
   verify.
