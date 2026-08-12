# Exercise 4 — GQA + KV-head replication under TP

**Goal**: extend ex03's MHA to Grouped Query Attention (GQA), and handle the
case `tp_size > num_kv_heads` by **replicating KV heads across ranks**.
This is the fix that's missing from `nanovllm-jun/nanovllm/layers/linear.py`
— it asserts `num_kv_heads % tp_size == 0` and dies at TP-8 on
Qwen3-30B-A3B (`num_kv_heads = 4`).

Real project landmark: whatever you write here drops into nanovllm-jun in
Week 2 with minimal changes.

## The math

Let $X \in \mathbb{R}^{B \times T \times H}$, $H_q$ Q heads, $H_{kv}$ KV
heads, and head dim $D$. In GQA, $H_{kv} \le H_q$ and $H_q$ is divisible
by $H_{kv}$ (Q heads are partitioned into $H_{kv}$ groups, each group
shares one KV head). Weights:

$$
W_q \in \mathbb{R}^{H_q D \times H}, \quad
W_k \in \mathbb{R}^{H_{kv} D \times H}, \quad
W_v \in \mathbb{R}^{H_{kv} D \times H}, \quad
W_o \in \mathbb{R}^{H \times H_q D}.
$$

Per-token attention: each Q head $h_q$ attends against KV head
$h_{kv}(h_q) = h_q \bmod H_{kv}$ (or `h_q // (H_q / H_kv)` — the two
formulations are equivalent for the "contiguous group" arrangement used
in practice). Q heads $\{0, 1, \dots, H_q/H_{kv} - 1\}$ share KV head 0,
etc.

Qwen3-30B-A3B: `H_q = 32`, `H_{kv} = 4`, `D = 128`. GQA ratio 8:1.

### Under TP-$N$

**Case A: $N \le H_{kv}$ (no replication needed)** — each rank owns
$H_q / N$ Q heads and $H_{kv} / N$ KV heads. Same as ex03, just with
smaller K/V sizes than Q.

**Case B: $N > H_{kv}$ (KV replication)** — there aren't enough KV
heads to give each rank one. Define $r_{kv} = N / H_{kv}$ replicas.
Every rank holds **1 KV head** (its assigned one), and groups of
$r_{kv}$ ranks share the same KV head.

Concretely for Qwen3 at TP-8 (`H_{kv}=4`, `N=8`, so $r_{kv}=2$):

| Rank | Q heads owned | KV head owned |
|---|---|---|
| 0 | 0-3   | 0 (shared with rank 1) |
| 1 | 4-7   | 0 (shared with rank 0) |
| 2 | 8-11  | 1 (shared with rank 3) |
| 3 | 12-15 | 1 (shared with rank 2) |
| 4 | 16-19 | 2 (shared with rank 5) |
| 5 | 20-23 | 2 (shared with rank 4) |
| 6 | 24-27 | 3 (shared with rank 7) |
| 7 | 28-31 | 3 (shared with rank 6) |

Each rank's KV head is exactly the one its Q heads need (Q heads 0-3
use KV head 0, and this rank owns KV head 0 — no cross-rank KV traffic
is ever needed).

### Per-rank storage

Let $n_q^{(r)} = H_q / N$ (Q heads per rank) and
$n_{kv}^{(r)} = \max(1,\, H_{kv} / N)$ (KV heads per rank; the `max` is
what triggers replication). The merged QKV weight per rank has shape

$$
W_{qkv}^{(r)} \;\in\; \mathbb{R}^{(n_q^{(r)} + 2 n_{kv}^{(r)}) D \;\times\; H}.
$$

Q shard: rows $[0, n_q^{(r)} D)$.
K shard: rows $[n_q^{(r)} D, (n_q^{(r)} + n_{kv}^{(r)}) D)$.
V shard: rows $[(n_q^{(r)} + n_{kv}^{(r)}) D, (n_q^{(r)} + 2 n_{kv}^{(r)}) D)$.

**Total across all ranks** = $N \cdot (n_q^{(r)} + 2 n_{kv}^{(r)}) D$
= $(H_q + 2 H_{kv} r_{kv}) D$. Note that in the replication case this is
*larger* than the un-sharded total $(H_q + 2 H_{kv}) D$ — because we're
storing $r_{kv}$ redundant copies of the KV weights. Memory cost, not
correctness cost.

### Attention path per rank

After the merged QKV GEMM, on each rank:

- $Q_r \in [B, T, n_q^{(r)}, D]$
- $K_r, V_r \in [B, T, n_{kv}^{(r)}, D]$   ($n_{kv}^{(r)} \le n_q^{(r)}$)

Apply RoPE to $Q_r, K_r$, then broadcast K/V up to match Q's head count
by **`repeat_interleave` on the head axis** with factor
$n_{rep} = n_q^{(r)} / n_{kv}^{(r)}$. After broadcasting, K and V have
shape $[B, T, n_q^{(r)}, D]$ and SDPA runs as usual.

The output projection $W_o$ is row-parallel with input dim
$n_q^{(r)} D$ per rank; one all-reduce at exit restores the replicated
$Y$ of shape $[B, T, H]$.

## Divisibility invariant

For our replication rule to be a clean partition, we require:

$$
H_q \bmod N = 0 \quad \text{AND} \quad \bigl(H_{kv} \bmod N = 0 \;\lor\; N \bmod H_{kv} = 0\bigr).
$$

Either $H_{kv}$ is a multiple of $N$ (normal sharding) or $N$ is a
multiple of $H_{kv}$ (integer replication). We assert this at construction.

For **Qwen3-30B-A3B** with $H_q=32, H_{kv}=4$, valid `tp_size` values are
$\{1, 2, 4, 8, 16, 32, \dots\}$ — every divisor of 32 that also relates
cleanly to 4. TP-3 or TP-5 would fail the assertion.

## Lifecycle — when each method is called

Same 4-phase pattern as ex01-03 (see earlier READMEs for the full
timeline). Ex04-specific notes:

- **Phase 1** (`__init__`): the `output_size` calculation now depends on
  the replication regime. `output_size` is *larger* than the sum of Q/K/V
  full weights when KV is replicated.
- **Phase 3** (`weight_loader`): called **3 times** per instance — once
  each for `"q"`, `"k"`, `"v"`. Under replication, multiple ranks apply
  the *same* slice of the K (or V) full weight into their local buffers.
- **Phase 4** (`forward`): includes a `repeat_interleave` on K and V
  after RoPE to broadcast to Q's per-rank head count. This is a
  *runtime* replication (viewed vs materialized depends on
  `repeat_interleave`'s implementation — for our purposes treat it as
  a broadcast helper that costs 0 memory in the ideal case).

## What to fill in

1. `QKVParallelLinearGQA.__init__`:
   - Compute `num_heads_per_rank`, `num_kv_heads_per_rank`,
     `num_kv_replicas`, `q_size_per_rank`, `kv_size_per_rank`.
   - Compute the merged `output_size` for the parent.
   - Call `super().__init__(hidden, output_size, tp_size, tp_rank, group=group)`.
2. `QKVParallelLinearGQA.weight_loader`:
   - Offset math identical to ex03 (Q at 0, K at q_size, V at q_size+kv_size).
   - Q slicing: `chunk(tp_size, 0)[tp_rank]`.
   - K/V slicing: `chunk(effective_chunks, 0)[effective_chunk_id]` where
     `effective_chunks = min(num_kv_heads, tp_size)` and
     `effective_chunk_id = tp_rank // num_kv_replicas`.
3. `TPGQA.__init__` / `forward`: mostly the same as ex03's `TPMHA`,
   with the split sizes reflecting `n_kv_heads_per_rank * head_dim`
   (smaller than Q's), and a `repeat_interleave` on K/V after RoPE.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex04_gqa_tp.py -v
```

The test parametrizes `tp_size ∈ {1, 2, 4, 8}` — the 8 case is the
KV-replication case; the others are normal sharding. Also parametrizes
`dtype ∈ {fp32, bf16}`. Total 8 test cases.

## Why this exercise matters for the project

This IS the fix for nanovllm-jun's attention TP path. Look at
`nanovllm-jun/nanovllm/models/qwen3.py::Qwen3Attention.__init__` — it
asserts `num_kv_heads % tp_size == 0`, which forbids TP-8 for Qwen3.
Your `QKVParallelLinearGQA` relaxes that assertion via replication.
Direct drop-in for the port back in Week 2.

## For your paper's Verus spec

Ex04's theorem generalizes ex03's:

$$
\forall\, N, H_q, H_{kv} \text{ with } N \mid H_q \text{ and } (H_{kv} \mid N \text{ or } N \mid H_{kv}):\\
\text{TPGQA}(N)(x) \;\equiv\; \text{RefGQA}(x) \quad \text{on replicated input } x.
$$

The proof case-splits on the two regimes (sharding vs replication), but
both cases share the same "concat-then-vstack = sum-of-per-rank-products"
identity from ex03's O-projection. The additional lemma for Ex04: **the
KV replicas hold identical bytes**, so any collective that averages or
selects one canonical copy across replicas is a no-op.
