# Ex01 — Formal properties for verification

This directory contains **attempt proofs** in three tools (Verus / Dafny / Z3)
of the correctness properties for `ColumnParallelLinear` and `RowParallelLinear`
from [../solution.py](../solution.py). It's the first exercise of the paper's
verification track — the scaffold for the pattern we'll replicate across Ex02
through Ex07.

The linear-algebra content (`x @ W.T` is a matrix multiplication) is **not**
what these tools reason about natively. They reason about **discrete
structural properties** — sharding, concatenation, weight loading, message
passing — with matmul treated as an uninterpreted function that satisfies
two declared axioms. The proof shape is:

> *Given* matmul's split axioms, TP-parallel linear layers preserve their
> semantic contract (sequential-equivalent output up to reduction order).

The linear-algebra axioms themselves are proved in mathematics, not in the
tool.

## Abstraction model

Common to all three proofs:

| Concept | Model |
|---|---|
| Element type | `int` (Verus/Dafny) or `Real` (Z3). Abstracted from `bf16`/`fp32`. |
| Tensor `[M, N]` | `Seq<Seq<Element>>` — a sequence of rows, each row a sequence of columns. |
| `full_weight` | A tensor of shape `[out_features, in_features]` — the unsharded reference. |
| `narrow(t, start, len)` | `t[start .. start+len]` — extract contiguous rows (or columns after transpose). |
| `concat(a, b)` | `a + b` — sequence concatenation, both dims possible. |
| `matmul(x, w_t)` | **Uninterpreted** binary spec function. See axioms below. |
| Rank | A natural number in `[0, tp_size)`. |

The row / column distinction is folded into the axis argument of `narrow` and
`concat` — most proofs work on dim-0 (row-major), and column-parallel /
row-parallel differ only in *which dim* the sharding acts on. See
[column_parallel.dfy](column_parallel.dfy) §1 for the concrete definitions.

### Matmul axioms

Two axioms about `matmul` are declared and used but not proved:

**Axiom M1 — matmul splits over the out-dim of the weight**:

$$
\forall x, W_0, W_1: \quad \mathrm{matmul}(x, (W_0 \Vert W_1)^\top) = \mathrm{matmul}(x, W_0^\top) \parallel \mathrm{matmul}(x, W_1^\top)
$$

where $\Vert$ concatenates along the out-dim of $W$ and $\parallel$
concatenates along the corresponding out-dim of the output.

**Axiom M2 — matmul splits over the in-dim of the weight with sum**:

$$
\forall x, W_0, W_1: \quad \mathrm{matmul}(x_0 \Vert x_1, (W_0 \Vert W_1)^\top) = \mathrm{matmul}(x_0, W_0^\top) + \mathrm{matmul}(x_1, W_1^\top)
$$

where the LHS's concatenation is on the in-dim of $W$ (and the corresponding
last dim of $x$), and the RHS's `+` is elementwise sum.

**These two axioms are the entire linear-algebra content of the proof.** M1
justifies Column-parallel (no comm — outputs sharded on out-dim). M2
justifies Row-parallel (all-reduce sum — combine partials over in-dim).
Every downstream tensor-parallel proof in the paper reduces to some
composition of M1 and M2 applied to specific layers.

The paper will state these as axioms in the abstract spec and cite the
matrix-multiplication distributivity theorem they follow from (standard
linear algebra, provable in about 5 lines by induction on tensor dims). The
proof artifact never re-proves them.

## Properties to verify — ColumnParallelLinear

Let `full_weight: [M, N]` with `M % tp_size == 0`, `shard_size := M // tp_size`.
Let `shard(r) := full_weight[r * shard_size .. (r+1) * shard_size, :]`.

### C1 — Shard-and-gather roundtrip

$$
\forall W, \text{tp\_size} : M \bmod \text{tp\_size} = 0 \implies
\bigparallel_{r=0}^{\text{tp\_size}-1} \mathrm{shard}(r) = W
$$

Concatenating shards along dim 0 reconstructs the full weight. **Corollary**:
no data is lost or duplicated in sharding.

### C2 — Sharding disjointness

$$
\forall r \ne r', i : \mathrm{index}(i, \mathrm{shard}(r)) \cap \mathrm{index}(i, \mathrm{shard}(r')) = \emptyset
$$

Any two distinct ranks hold disjoint slices of `full_weight`. **Corollary**:
sharding is a *partition*, not a covering.

### C3 — Weight loader post-condition

After `ColumnParallelLinear(rank=r).weight_loader(full_weight)` returns:

$$
\text{self.weight.data} = \mathrm{shard}(r) = \mathrm{narrow}(W, r \cdot \text{shard\_size}, \text{shard\_size})
$$

Direct implementation-conforming assertion. This is the "Phase 3 post-condition"
from the 4-phase lifecycle in [../README.md](../README.md).

### C4 — Forward correctness (via axiom M1)

Let $y_r := \mathrm{ColumnParallelLinear}(r).\mathrm{forward}(x) = \mathrm{matmul}(x, \mathrm{shard}(r)^\top)$.
Then:

$$
\bigparallel_{r=0}^{\text{tp\_size}-1} y_r = \mathrm{matmul}(x, W^\top)
$$

i.e. gathering the per-rank outputs on the last dim reconstructs the full,
unsharded forward output.

**Proof sketch**: expand $y_r$ using axiom M1 unfolded $\text{tp\_size} - 1$
times over the shards. Each unfolding preserves the concatenation structure.
Base case: $\text{tp\_size} = 1$ gives $y_0 = \mathrm{matmul}(x, W^\top)$
trivially.

## Properties to verify — RowParallelLinear

Let `full_weight: [M, N]` with `N % tp_size == 0`, `shard_size := N // tp_size`.
Let `shard(r) := full_weight[:, r * shard_size .. (r+1) * shard_size]`.
Let `x_shard(r) := x[:, r * shard_size .. (r+1) * shard_size]`.

### R1 — Shard-and-gather roundtrip (dim=1)

$$
\forall W, \text{tp\_size} : N \bmod \text{tp\_size} = 0 \implies
\bigparallel^1_{r=0}^{\text{tp\_size}-1} \mathrm{shard}(r) = W
$$

The dim-1 analog of C1.

### R2 — Sharding disjointness (dim=1)

Same as C2 but on the in-dim.

### R3 — Weight loader post-condition

$$
\text{self.weight.data} = \mathrm{shard}(r) = \mathrm{narrow\_dim1}(W, r \cdot \text{shard\_size}, \text{shard\_size})
$$

### R4 — All-reduce correctness (via axiom M2)

Let $y_r^{\text{partial}} := \mathrm{matmul}(x_{\text{shard}}(r), \mathrm{shard}(r)^\top)$
(the pre-all-reduce partial product). Then:

$$
\sum_{r=0}^{\text{tp\_size}-1} y_r^{\text{partial}} = \mathrm{matmul}(x, W^\top)
$$

The all-reduce sum reconstructs the full matmul.

**Proof sketch**: repeated application of axiom M2. `all_reduce(SUM)` is
modeled as elementwise sum of the tensors across ranks — the paper's abstract
spec treats it as an atomic collective event.

## Composition property — Column → Row

Two layers chained together (as in Ex02's SwiGLU MLP):

$$
\begin{aligned}
z_r &:= \mathrm{ColumnParallelLinear}(r).\mathrm{forward}(x)     && \text{shape } [B, N/\text{tp\_size}]\\
y_r &:= \mathrm{RowParallelLinear}(r).\mathrm{forward}(z_r)      && \text{shape } [B, N], \text{ replicated}
\end{aligned}
$$

**Property**: $y_r$ is *replicated* across ranks and equals
$\mathrm{matmul}(\mathrm{matmul}(x, W_1^\top), W_2^\top)$ — the unsharded
two-layer product. **One all-reduce total** (inside the row parallel).

**Proof**: chain C4 with R4. Column-parallel's output shards on dim 1 (the
in-dim of the next layer) exactly match Row-parallel's expected input
sharding, so no intermediate collective is needed.

## What each tool actually proves

The three attempts operate at different points of the "coverage vs. cost" curve.

| Tool | Approach | What it proves | What it doesn't |
|---|---|---|---|
| **Dafny** | Universal quantifiers over `nat` and `seq<T>` | C1, C2, C3 fully (parameterized over `tp_size` and `M`); C4 up to axiom M1 | Nothing about `int` overflow or fixed-precision numerics |
| **Verus** | Same as Dafny but Rust-hosted, more Rustic syntax | Same coverage as Dafny in principle; may need more `assert(...)` hints in practice | Same as Dafny |
| **Z3** | Bounded model check via Python `z3-solver` | C1–C4 for specific `(tp_size, M, N)` concrete values (e.g., `tp_size ∈ {2, 4, 8}`, `M = 32`, `N = 16`) | No general parameterized guarantee — verifies only the cases enumerated |

**Interpretation for the paper**: Dafny/Verus give the *parameterized* theorem
(the paper's headline claim). Z3 gives *concrete regression evidence* — a
suite of tp_size × shape combinations that we've mechanically confirmed match
the theorem. Concrete regressions complement the parameterized proof and
catch mistakes in the axiom encoding: if the axiomatic version admits some
`tp_size = 3` counterexample that the concrete Z3 check doesn't reproduce,
one of them is wrong.

## Files in this directory

| File | Content |
|---|---|
| [PROPERTIES.md](PROPERTIES.md) | This document — properties + axioms + tool coverage |
| [column_parallel.dfy](column_parallel.dfy) | Dafny attempt at properties C1–C4 |
| [column_parallel.rs](column_parallel.rs) | Verus attempt at properties C1–C4 |
| [column_parallel_z3.py](column_parallel_z3.py) | Z3 bounded verification for `tp_size ∈ {1, 2, 4, 8}`, shape `M×N = 32×16` |
| [README.md](README.md) | How to install each tool and run each proof |

RowParallelLinear (properties R1–R4) is a natural follow-up — the sharding
axis flips from dim 0 to dim 1, and matmul-axiom M1 is replaced with M2.
Adding it will roughly duplicate this scaffold with axis-parameterized
functions; see [README.md](README.md) for the delivery plan.

## Correctness of the abstraction

Two questions the paper's soundness section must answer:

1. **Are the matmul axioms M1 and M2 true?** Yes — they follow from
   distributivity of matrix multiplication over row/column concatenation.
   Cite [Horn & Johnson §0.4] or any linear algebra text.
2. **Does our abstract model (`Seq<Seq<int>>` + uninterpreted matmul) faithfully
   represent the Python code?** This is a *refinement claim*. The paper
   argues: (a) PyTorch tensors are backed by row-major memory (so
   `Seq<Seq<...>>` is faithful up to layout); (b) `F.linear(x, W)` is a
   correct implementation of `matmul(x, W^T)` (given by PyTorch's spec); and
   (c) `.narrow(0, r*s, s)` on a `[M, N]` tensor returns the same slice as
   `w[r*s : (r+1)*s]` in the abstract model. These are standard refinement
   arguments — the paper will state them briefly and move on.

The verification content is (1) plus (a) + (b) + (c). The tool proofs
handle everything else.
