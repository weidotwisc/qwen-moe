# Ex09 — Formal properties for the fused-MoE Triton grouped-GEMM kernel

This directory formalizes the exact-arithmetic DSL semantics of the three
grouped-GEMM launches in [`reference.py`](../reference.py): gate projection,
up projection, and down projection around the pointwise SwiGLU activation.

The machine-checked theorem compares this DSL model with the mathematical
Python per-expert reference. It does not verify Triton-to-PTX compilation,
hardware execution, floating-point rounding, or parse the Python/Triton source
into Verus. Source-to-model correspondence remains an explicit audit boundary.

## Model

`fused_kernel_dsl.rs` represents tensors as exact integer matrices. For one
output element, the Python reference computes the untiled dot product

$$
\sum_{k=0}^{K-1} x_k w_k,
$$

while the Triton model partitions the same products into `BLOCK_K` intervals,
accumulates the intervals in program order, and clamps the last interval to
`K`. Clamping is the exact-arithmetic meaning of Triton's zero padding.

The dispatch-table contract states that every logical row is covered by
exactly one M-axis program and that the program selects the same expert as the
offsets-based Python owner function. The N-axis launch grid covers every output
column exactly once.

## Machine-checked properties

### K1 — Output-tile coverage and disjointness

- `k1a_m_axis_disjoint` proves that distinct M-axis programs cannot cover the
  same logical output row under the dispatch contract.
- `k1b_n_axis_covers` proves that the N-axis launch grid covers every logical
  output column.
- `k1c_n_axis_disjoint` proves that distinct N-axis program identifiers have
  disjoint column intervals.

### K2 — Exact K-reduction correctness

`k2_k_reduce_correctness` proves by induction that the sequence of clamped
`BLOCK_K` reductions equals the untiled dot product. The proof uses concrete
integer multiplication and addition; it has no matmul-splitting or
zero-padding axiom.

### K3 — Grouped-matmul correctness

`k3_grouped_matmul_dsl_equals_reference` lifts K2 to every output row and
column. Under the dispatch and launch-shape contracts, the modeled Triton
grouped GEMM equals the mathematical Python grouped-matmul reference.

### F4 — Complete fused-expert correctness

`f4_fused_moe_dsl_equals_python_reference` composes K3 for gate, up, and down
projections. Since both paths apply the same deterministic pointwise
`silu(gate) * up` operation, the complete modeled fused expert equals the
Python expert reference in exact arithmetic.

`component_integration.rs::ex09_dsl_execution_establishes_postcondition`
imports this result into the shared MoE composition model. Consequently,
Ex09's pointwise output postcondition is derived rather than assumed by the
composition theorem.

## Remaining trust boundary

The composition contract still records that the concrete Ex09 run corresponds
to `fused_moe_dsl`, and that the mathematical Python reference represents the
shared abstract `expert_apply`. Auditing these relations requires checking:

- `_build_tile_dispatch` against `dispatch_matches_row_owner`;
- `grouped_matmul_kernel_v2` against the modeled M/N/K indexing and masking;
- the wrapper's gate/up/SwiGLU/down call sequence against `fused_moe_dsl`;
- the compiler and GPU execution stack; and
- numerical behavior separately, because the theorem uses exact integers.

The runtime tests in `bootcamp/tests/test_ex09_fused_moe.py` provide empirical
evidence for these implementation boundaries; they are not part of the formal
proof.
