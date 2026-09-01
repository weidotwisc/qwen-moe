# Ex03 — Formal properties for multi-head attention under TP

This directory formalizes the correctness of
[`solution.py`](../solution.py) — the `QKVParallelLinear` + `TPMHA`
composition implementing multi-head attention with tensor parallelism.
Structurally analogous to Ex02: a *three-way* merged column-parallel linear
(Q + K + V, versus Ex02's two-way gate + up), followed by attention on the
sharded heads, followed by a row-parallel output projection.

**Style follows [ex02/verification/PROPERTIES.md](../../ex02_mlp_tp/verification/PROPERTIES.md).**
The abstraction model (`Tensor = Seq<Row>`, matmul uninterpreted, axiom M1)
extends Ex02 with two new elements: a **three-way merged shard** and an
**attention** uninterpreted op with a "attention commutes with head-shard"
axiom.

## Abstraction model additions

Ex02's abstractions are inherited unchanged. Two new elements:

| Concept | Model |
|---|---|
| `qkv_shard(w_q, w_k, w_v, r, tp)` | Three-way concat on dim 0: `shard(w_q, r, tp) + shard(w_k, r, tp) + shard(w_v, r, tp)` — this rank's packed (Q, K, V) slice. |
| `attention(q, k, v)` | Uninterpreted ternary op modeling per-head scaled-dot-product attention (softmax + matmul with causal mask included). |

`attention` is uninterpreted because scaled-dot-product attention involves
softmax, which is a non-linear op that isn't part of the "schedule
correctness" content of the proof. Its shape and shard-commuting behavior
are all the proof needs — captured by **axiom A1** below.

**Axiom A1 — attention commutes with head-shard**:

$$
\forall q, k, v : \quad \mathrm{attention}(\mathrm{concat}_1(q_0, q_1), \mathrm{concat}_1(k_0, k_1), \mathrm{concat}_1(v_0, v_1))
= \mathrm{concat}_1(\mathrm{attention}(q_0, k_0, v_0), \mathrm{attention}(q_1, k_1, v_1))
$$

where `concat_1` concatenates along the head-axis (dim 1 in the
`[B*T, n_heads, head_dim]` view; equivalently dim 0 in our
`Tensor = Seq<Row>` model where each row is a single head's data). This
captures the semantic fact that attention is head-local: attention on
concatenated head-shards equals concatenation of per-head-shard attentions.
It's the reason MHA is TP-parallelizable at all.

## Properties to verify — QKVParallelLinear

Let `w_q: [H*D, hidden]`, `w_k: [H*D, hidden]`, `w_v: [H*D, hidden]` where
`H = num_heads`, `D = head_dim`, with `H % tp_size == 0`. Let
`q_shard(r) := shard(w_q, r, tp_size)`, similarly for `k_shard(r)`, `v_shard(r)`.
Let `qkv_full := w_q + w_k + w_v` (concat on dim 0) — the merged full weight.
Let `qkv_shard(r) := q_shard(r) + k_shard(r) + v_shard(r)`.

### Q1 — Per-projection weight_loader post-condition

After calling `weight_loader(w_q, "q")`, `weight_loader(w_k, "k")`, and
`weight_loader(w_v, "v")` on rank `r`:

$$
\text{self.weight.data} = \mathrm{qkv\_shard}(r)
$$

Analogous to Ex02's M1 but for three projections. **Proof**: definitional
post-condition of the three narrow-copy calls into disjoint regions.

### Q2 — QKV shards partition qkv_full (weakened form)

The three separate gather operations reconstruct `w_q`, `w_k`, `w_v`
independently:

$$
\bigparallel_r q\_\mathrm{shard}(r) = w_q, \quad
\bigparallel_r k\_\mathrm{shard}(r) = w_k, \quad
\bigparallel_r v\_\mathrm{shard}(r) = w_v
$$

**Proof**: three applications of ex01's `c1_shard_gather_roundtrip`.

### Q3 — Merged QKV forward correctness (via axiom M1, tp=2)

The QKV projection on rank `r` is
`qkv_out_r := matmul(x, transpose(qkv_shard(r)))`. Gathering:

$$
\bigparallel_{r=0}^{\text{tp}-1} qkv\_out_r = \mathrm{matmul}(x, \mathrm{transpose}(qkv\_full))
$$

**Proof**: apply Ex02's M1 axiom twice (three-way concat = two nested
binary concats). Stated at tp=2 concretely; parameterized version stubbed.

### Q4 — Three-way split matches per-projection shards

For any rank `r`, `qkv_out_r` splits on dim 1 into three head-shard-sized
regions:
- $q_r = \mathrm{matmul}(x, \mathrm{transpose}(q\_\mathrm{shard}(r)))$
- $k_r = \mathrm{matmul}(x, \mathrm{transpose}(k\_\mathrm{shard}(r)))$
- $v_r = \mathrm{matmul}(x, \mathrm{transpose}(v\_\mathrm{shard}(r)))$

**Proof**: three-way application of axiom M1. This is what makes the
Python `torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)` correspond to
the abstract split.

## Properties to verify — TPMHA composition

Let $q_r$, $k_r$, $v_r$ be as in Q4. Define
$a_r := \mathrm{attention}(q_r, k_r, v_r)$ — this rank's local attention
output on its head shard. Let $w_o: [\mathrm{hidden}, H \cdot D]$ with
$H \cdot D \bmod \text{tp\_size} = 0$; let $w_o^{\mathrm{shard}}(r)$ be its
dim-1 shard (RowParallelLinear).

The block output is:

$$
y := \sum_{r=0}^{\text{tp}-1} \mathrm{matmul}(a_r, \mathrm{transpose}(w_o^{\mathrm{shard}}(r)))
$$

### A1 — Attention commutes with head-gather

$$
\bigparallel_r a_r = \bigparallel_r \mathrm{attention}(q_r, k_r, v_r)
= \mathrm{attention}(\bigparallel_r q_r, \bigparallel_r k_r, \bigparallel_r v_r)
$$

**Proof**: axiom A1 (attention head-local property).

### A2 — Block correctness (via axioms M1, M2, A1)

$y$ equals the unsharded MHA output:

$$
y = \mathrm{matmul}(\mathrm{attention}(\mathrm{matmul}(x, W_q^\top), \mathrm{matmul}(x, W_k^\top), \mathrm{matmul}(x, W_v^\top)), W_o^\top)
$$

**Proof**: chain Q3 (gathered QKV = unsharded matmul) with A1 (attention
commutes with gather) with ex01's R4 (row-parallel all-reduce sum
reconstructs the o_proj matmul).

## What each tool proves — this exercise

Same three-tool pattern as Ex01/Ex02. Verus proof shipped first. See
[ex01's README](../../ex01_linear_tp/verification/README.md) for the
tool-comparison table.

## Correspondence to Python

The proof abstracts `qkv_shard`, `attention`, and the two `matmul`
projections. Correspondence to Python:

1. `QKVParallelLinear.weight_loader(w_q, "q")`, `(w_k, "k")`, `(w_v, "v")`
   populate the fused weight identically to a plain
   `ColumnParallelLinear` loaded with `w_q + w_k + w_v`. Verified by Q1.
2. `TPMHA.forward(x)` does
   `o_proj(SDPA(rope(q), rope(k), v))` where `q, k, v` come from
   `split(qkv_proj(x), [q_size, kv_size, kv_size], dim=-1)`. The Verus
   proof abstracts RoPE and SDPA into the single `attention` op — this is
   sound because RoPE is applied element-wise per head (head-local) and
   SDPA is head-local by construction. See §"Correctness of the
   abstraction" below for the refinement argument.
3. The `torch.split` in Python is the "three-way split of `qkv_out_r`" in
   Q4.

## Correctness of the abstraction

Two questions:

1. **Is axiom A1 (attention commutes with head-shard) true?** Yes. SDPA is
   defined as `softmax((Q @ K^T) / sqrt(d)) @ V`, and each of `Q @ K^T`,
   the softmax, and the final matmul acts *per-head* on the head-major
   layout `[B, n_heads, T, head_dim]`. Attention on a subset of heads is
   the corresponding subset of attention outputs. This is a standard
   linear-algebra fact about the SDPA definition.

2. **Does folding RoPE into `attention` preserve correctness?** Yes. RoPE
   is defined element-wise per head (the rotary factor depends only on
   position and dimension index within a head, not on other heads). So
   `rope(q_r)` on head-shard `r` equals the corresponding head-slice of
   `rope(q)` on the full tensor. Verus's `attention` op is defined as
   "`attention(rope(q), rope(k), v)`" — the fact that RoPE commutes with
   head-shard is folded into A1's semantics.
