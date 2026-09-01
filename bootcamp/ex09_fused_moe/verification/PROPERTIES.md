# Ex09 — Formal properties for the fused-MoE Triton grouped-GEMM kernel

This directory formalizes the correctness of
[`solution.py`](../solution.py) — the Triton kernel that replaces the
per-expert Python loop in Ex05b/Ex06/Ex07 with a single grouped-GEMM
launch.

**The Triton kernel is NOT verified directly.** GPU kernels are outside
the scope of source-level formal methods; instead we verify the
**algorithmic contract** of the kernel — the pre/post condition that the
kernel promises to satisfy — and use that contract as the target that
downstream compositions (Ex10 fused hybrid) refine.

**Style follows [ex06/verification/PROPERTIES.md](../../ex06_ep_pure/verification/PROPERTIES.md).**

## Abstraction model

The kernel is treated as an uninterpreted spec function:

$$
\mathrm{fused\_moe\_forward}(\text{sorted\_x}, \text{offsets}, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}) \to \text{out}
$$

Precondition (formalized in Verus as `fused_moe_precondition`):
- `sorted_x: [M, H]`, well-formed rows all length H.
- `offsets: [E+1]` with `offsets[0] == 0` and monotone non-decreasing.
- `offsets[E] == M` (token conservation, from Ex06).
- `W_gate, W_up: [E, I, H]`; `W_down: [E, H, I]`; all well-formed.

Postcondition (formalized as `fused_moe_postcondition`):
- `out: [M, H]`, well-formed.
- For every row `i` in `[0, M)`, letting `e` be the unique expert with
  `offsets[e] <= i < offsets[e+1]`:
  `out[i] == expert_apply(e, sorted_x[i])`
  up to declared floating-point tolerance.

Where `expert_apply(e, x)` is the semantic per-expert operation
`down_proj(silu(gate_proj(x)) * up_proj(x))`, uninterpreted at this level.

## Properties to verify

### F1 — Precondition consistency

The precondition is internally consistent:
- Monotonic offsets + `offsets[0] == 0` + `offsets[E] == M` implies
  every expert's block `[offsets[e], offsets[e+1])` is contained in
  `[0, M)`.
- Every row index in `[0, M)` falls in exactly one expert's block.

**Proof**: from monotonicity + endpoint conditions.

### F2 — Postcondition determines output uniquely

Two calls to `fused_moe_forward` with the same
`(sorted_x, offsets, W_gate, W_up, W_down)` produce outputs that are
`approx_eq` to each other. (Deterministic-up-to-tolerance property.)

**Proof**: the postcondition specifies the output content pointwise, up
to tolerance; any two outputs both satisfying it are `approx_eq`.

### F3 — Empty-expert handling

If `offsets[e] == offsets[e+1]` for some expert `e`, expert `e` gets
zero tokens. The output at other tokens is unaffected — the empty expert
does not read or write outside its (zero-length) block.

**Proof**: from the postcondition — no row index falls in an empty
block, so no output row references expert `e`.

### F4 — Composition with Ex06/Ex07 (external stub)

If Ex06's or Ex07's dispatch step establishes the routing-conservation
precondition (i.e., produces `(sorted_x, offsets)` satisfying F1), then
calling `fused_moe_forward` on those inputs produces the same output
(up to `approx_eq`) as running Ex05b's per-expert Python loop.

**Proof composition**:
1. Ex06/Ex07's dispatch postcondition matches F1 (routing conservation).
2. F2 gives that the output is uniquely determined by inputs.
3. The Python per-expert loop refines the same postcondition (F3-style
   per-block computation).
4. By transitivity, fused kernel output equals Python-loop output up to
   `approx_eq`.

This is the load-bearing property for **Ex10's fused-hybrid composition
theorem**.

## What each tool proves — this exercise

Same three-tool pattern as Ex01-07. Verus proof shipped first.

## Correspondence to Python + Triton

1. The Python wrapper `fused_moe_forward` in `solution.py` calls the
   Triton kernel three times (gate, up, down projections). The final
   output is the down-projection result at the tokens' original
   positions in `sorted_x`.
2. The Triton kernel itself (`grouped_matmul_kernel`) is treated as
   uninterpreted; the correspondence between the kernel's SASS-level
   behavior and the abstract spec function is established by:
   - The unit tests in `bootcamp/tests/test_ex09_fused_moe.py`, which
     compare the kernel's output to a pure-PyTorch reference oracle at
     fp32 and bf16 tolerances.
   - The kernel's own comment invariants (see the module docstring in
     `solution.py`).

This is the **trust boundary** of the Ex09 verification: the Triton
kernel's per-block-tile behavior is trusted; the abstract algorithmic
contract is what downstream compositions rely on.

## Correctness of the abstraction

The uninterpreted `fused_moe_forward` spec function is an abstract
interface, not a claim about the Triton kernel's specific implementation.
The paper's threats-to-validity section names this trust surface:
GPU-kernel-level correctness of the Triton implementation is out of
scope for source-level formal verification tools like Verus, Dafny, and
Z3.
