# Ex02 — Formal properties for the SwiGLU-MLP TP block

This directory formalizes the correctness of
[`solution.py`](../solution.py) — the two-class composition
`MergedColumnParallelLinear` + `RowParallelLinear` that implements one SwiGLU
MLP block under tensor parallelism. Two things get proved: **(M1-M6)** the
merged-column layer preserves the same structural invariants as ex01's
`ColumnParallelLinear` while shard-loading two projections independently; and
**(T1-T3)** the composition of merged-column → SiLU × mul → row-parallel
reconstructs the full unsharded block up to axiom M1 (column split) and
axiom M2 (row split with sum).

**Style follows [ex01/verification/PROPERTIES.md](../../ex01_linear_tp/verification/PROPERTIES.md).**
The abstraction model (`Tensor = Seq<Row>`, matmul uninterpreted, axioms
M1/M2) is identical; only the properties are new.

## Abstraction model additions

Ex02 uses ex01's `Tensor` / `shard` / `matmul` / `transpose` unchanged. Two
additions:

| Concept | Model |
|---|---|
| `merged_shard(w0, w1, r, tp)` | Concatenation on dim 0 of `shard(w0, r, tp) + shard(w1, r, tp)` — this rank's merged (gate, up) slice. |
| `silu_mul(gate, up)` | Uninterpreted binary op modeling elementwise `SiLU(gate) * up`. |

`silu_mul` is uninterpreted for the same reason `matmul` is: SiLU × mul is
pointwise linear-algebra content, not a schedule property. Its axiomatic
property (**axiom S1**, below) is all the proof needs.

**Axiom S1 — elementwise commutes with column-sharding**:

$$
\mathrm{silu\_mul}(\mathrm{concat}(g_0, g_1), \mathrm{concat}(u_0, u_1))
= \mathrm{concat}(\mathrm{silu\_mul}(g_0, u_0), \mathrm{silu\_mul}(g_1, u_1))
$$

where `concat` is on dim 1 (the out-dim of gate/up). This axiom captures the
key semantic fact that makes TP-sharded SiLU × mul equal to the unsharded
version: **elementwise ops are shard-agnostic**.

## Properties to verify — MergedColumnParallelLinear

Let `w_gate: [I, H]`, `w_up: [I, H]` with `I % tp_size == 0`. Let
`gate_shard(r) := shard(w_gate, r, tp_size)`, similarly `up_shard(r)`. Let
`merged_full := w_gate + w_up` (concat on dim 0), so
`merged_full: [2*I, H]`. Let
`merged_shard(r) := gate_shard(r) + up_shard(r)`, the concatenation on dim 0
that this rank stores.

### M1 — Per-projection weight_loader post-condition

After calling `weight_loader(w_gate, shard_id=0)` and
`weight_loader(w_up, shard_id=1)` on rank `r`:

$$
\text{self.weight.data} = \mathrm{merged\_shard}(r)
$$

i.e., the two per-projection loads leave the fused weight matrix in exactly
the same state as loading `merged_full` in one shot into a plain
`ColumnParallelLinear(2*I, H)`.

### M2 — Merged shards partition merged_full

Gathering `merged_shard(r)` over `r` reconstructs `merged_full`:

$$
\bigparallel_{r=0}^{\text{tp}-1} \mathrm{merged\_shard}(r) = \mathrm{merged\_full}
$$

**Proof**: follows from ex01's `c1_shard_gather_roundtrip` applied to
`merged_full` (which has `|merged_full| == 2 * I`, still divisible by
`tp_size`).

### M3 — Merged forward correctness (via axiom M1)

The forward on rank `r` is
`y_r := matmul(x, transpose(merged_shard(r)))`. Gathering:

$$
\bigparallel_{r=0}^{\text{tp}-1} y_r = \mathrm{matmul}(x, \mathrm{transpose}(\mathrm{merged\_full}))
$$

**Proof**: identical to ex01's C4, applied to `merged_full`.

## Property to verify — TPSwiGLUMLP composition

Let $z_r := \mathrm{matmul}(x, \mathrm{transpose}(\mathrm{merged\_shard}(r)))$
be rank $r$'s output from the fused gate+up projection. Split
$z_r = (g_r, u_r)$ on dim 1 (this is the Python `gate_up.chunk(2, dim=-1)`).
Let $h_r := \mathrm{silu\_mul}(g_r, u_r)$ — the elementwise op. Let
$w_{\mathrm{down}}: [H, I]$ with $I \bmod \text{tp\_size} = 0$; let
$w_{\mathrm{down}}^{\mathrm{shard}}(r)$ be its dim-1 shard as in ex01's
`RowParallelLinear`.

The block output is:

$$
y := \sum_{r=0}^{\text{tp}-1} \mathrm{matmul}(h_r, \mathrm{transpose}(w_{\mathrm{down}}^{\mathrm{shard}}(r)))
$$

(the one all-reduce at the end).

### T1 — Chunk-split-matches-shard invariant

For any rank $r$, $z_r$ splits on dim 1 as $z_r = (g_r, u_r)$ where
$g_r = \mathrm{matmul}(x, \mathrm{transpose}(\mathrm{gate\_shard}(r)))$ and
$u_r = \mathrm{matmul}(x, \mathrm{transpose}(\mathrm{up\_shard}(r)))$.

**Proof**: application of axiom M1 to the two-way split of `merged_shard(r)`
into `gate_shard(r) + up_shard(r)`.

### T2 — Elementwise-commutes-with-gather

$$
\bigparallel_{r=0}^{\text{tp}-1} h_r
= \bigparallel_{r=0}^{\text{tp}-1} \mathrm{silu\_mul}(g_r, u_r)
= \mathrm{silu\_mul}(\bigparallel g_r, \bigparallel u_r)
= \mathrm{silu\_mul}(\mathrm{matmul}(x, W_{\mathrm{gate}}^\top), \mathrm{matmul}(x, W_{\mathrm{up}}^\top))
$$

**Proof**: axiom S1 (elementwise commutes with concat) plus M3 applied to
`w_gate` and `w_up` independently.

### T3 — Block correctness (via axioms M1, M2, S1)

$y$ equals the unsharded MLP output:

$$
y = \mathrm{matmul}(\mathrm{silu\_mul}(\mathrm{matmul}(x, W_{\mathrm{gate}}^\top), \mathrm{matmul}(x, W_{\mathrm{up}}^\top)), W_{\mathrm{down}}^\top)
$$

**Proof**: T2 provides the "concatenated-hidden equals unsharded-hidden"
step; ex01's R4 (all-reduce sum reconstructs the row-parallel matmul)
finishes.

## What each tool proves in this exercise

Same three-tool structure as Ex01: Verus / Dafny / Z3 (Verus proof shipped
first). See [ex01's README](../../ex01_linear_tp/verification/README.md) for
the tool-comparison table; the interpretation is the same.

## Correspondence to Python

The Verus proof abstracts `merged_shard` and `silu_mul`; the correspondence
to the Python code is:

1. `MergedColumnParallelLinear.weight_loader(w_i, shard_id=i)` for `i ∈ {0, 1}` populates
   the fused weight tensor identically to `ColumnParallelLinear.weight_loader(w_0 + w_1)`.
   Verified by `M1`.
2. `TPSwiGLUMLP.forward(x)` computes
   `down_proj(silu(gate_up_proj(x)[gate]) * gate_up_proj(x)[up])`, which
   Verus abstracts as the `matmul → silu_mul → matmul` chain above.
3. The `chunk(2, dim=-1)` split in Python is the "split $z_r = (g_r, u_r)$
   on dim 1" step in T1.
