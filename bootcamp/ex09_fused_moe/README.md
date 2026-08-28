# Ex09 — Fused MoE Triton kernel (forward-only)

Port Ex05b's per-expert compute to a Triton grouped-GEMM kernel. Forward
only — inference-shape, no autograd, no atomics.

## Goal

Replace Ex05b's Phase 6 (per-expert Python loop) with a Triton kernel that
does the same math per-expert but with tiled grouped-GEMM. The rest of
the pipeline (router, sort, dispatch, combine, all_gather) stays in PyTorch.

## Scope boundaries

**IN scope:**
- Per-expert grouped-GEMM tiling.
- SwiGLU: gate + up + silu×mul + down.
- Empty-expert skip (`offsets[e] == offsets[e+1]`).
- Correctness against a PyTorch reference at Qwen3-30B-A3B dims.

**OUT of scope (kept in PyTorch outside the kernel):**
- Router (gate + topk + softmax + norm_topk_prob) — cheap.
- Weight-multiply by routing weights — done AFTER kernel.
- `index_add_` combine (un-permute) — done AFTER kernel.
- Dispatch/combine collectives — Ex06/Ex07 layer.

This factoring matches vLLM's `FusedMoE` boundary exactly. It keeps the
kernel atomics-free — each expert's output block is a disjoint slice of
`sorted_out`.

## Interface

```python
sorted_out = fused_moe_forward(
    sorted_x,   # [Nk, H]              post-dispatch, sorted by expert id
    offsets,    # [E_per_rank + 1]     int64, per-expert row boundaries
    W_gate,     # [E_per_rank, I, H]   packed per-expert gate projection
    W_up,       # [E_per_rank, I, H]   packed per-expert up projection
    W_down,     # [E_per_rank, H, I]   packed per-expert down projection
)
# returns sorted_out [Nk, H]
```

For each expert `e ∈ [0, E)` and its row range `s = offsets[e], f = offsets[e+1]`:

```
x_chunk = sorted_x[s:f]                     # [n_e, H]
gate    = x_chunk @ W_gate[e].T             # [n_e, I]
up      = x_chunk @ W_up[e].T               # [n_e, I]
hid     = silu(gate) * up                   # [n_e, I]
out     = hid @ W_down[e].T                 # [n_e, H]
sorted_out[s:f] = out
```

## Design progression — three options

The exercise walks through three fusion levels. **Start with A, refactor
to C**. Option B is a research stretch, not required.

### Option A — three grouped-GEMM launches (start here)

Three separate kernel invocations, one per projection:

```
launch grouped_matmul_kernel(sorted_x, W_gate, gate_out, offsets, ACT=silu)
launch grouped_matmul_kernel(sorted_x, W_up,   up_out,   offsets, ACT=none)
hid = gate_out * up_out                          # elementwise, torch
launch grouped_matmul_kernel(hid, W_down, sorted_out, offsets, ACT=none)
```

Actually, cleaner: apply SiLU inside the gate launch OR outside. Both work.
Simpler to leave activations OUT of the kernel and do `hid = silu(gate_out) * up_out`
in PyTorch. That matches the reference.

**Pros**: One kernel body, invoked 3×. Easiest to write correctly.
**Cons**: 3 kernel launches. Gate and up each reload `sorted_x`
independently → 2× HBM traffic on the input side.

### Option C — fused gate+up + separate down (production-shape)

vLLM's real pattern. One kernel produces `gate_up_out [Nk, 2I]` by reading
`sorted_x` once and doing paired-column matmul against `[W_gate | W_up]`.
Then a small elementwise op does `silu(gate) * up`. Then the down kernel
(same as Option A).

**Pros**: 2 launches. `sorted_x` loaded once for gate+up.
**Cons**: Kernel has two accumulators. Weight-packing gets a joint
`W_gate_up [E, 2I, H]` layout (or two pointer args).

### Option B — fully fused (research stretch, not required)

One kernel does gate+up+silu×mul+down in a single launch. Requires
handling TWO nested reductions (over H then over I) with the intermediate
`[BLOCK_M, BLOCK_I]` tiled in registers or SRAM. This is Megablocks-style;
manually writing it is rare in production because compilers
(`torch.compile` / inductor) generate B-style fused kernels automatically
at competitive quality.

Why we skip B by default: complexity/perf trade-off isn't favorable at
Qwen3-30B-A3B dims. The `[Nk, I]` intermediate is ~15 MB bf16 —
HBM round-trip is ~10 μs vs ~1-2 ms of matmul compute. Fusing away
that 10 μs isn't worth the nested-reduction complexity for the paper /
interview goal.

## Kernel signature (Option A)

```python
@triton.jit
def grouped_matmul_kernel(
    a_ptr,           # [M, K]      input
    b_ptr,           # [E, N, K]   per-expert weight (row-major)
    c_ptr,           # [M, N]      output
    offsets_ptr,     # [E+1]       int64 row boundaries
    M, N, K, E,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,  # 0=none, 1=silu
):
```

Grid: `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`.

Each program (pid_m, pid_n) computes one tile of `c[m_start:m_start+BLOCK_M, n_start:n_start+BLOCK_N]`.

## Grid + expert-lookup design

Two approaches to determining "which expert does this M-tile belong to":

### Option i — binary search on offsets (simple, works out of the box)

Each kernel program does a small binary search on `offsets_ptr` to find
its expert `e`. `E ≤ 128` at Qwen3 scale, so log₂(128) = 7 iterations —
negligible overhead. If the tile spans expert boundaries (i.e., some rows
belong to expert e and some to e+1), you either:
- Pad each expert's block to `BLOCK_M` rows so tiles never straddle
  (vLLM's `num_tokens_post_padded` trick — recommended), OR
- Mask off the wrong-expert rows in the tile.

Recommended: **padding at wrapper level**. Add a Python-side helper that
computes padded offsets: `padded_offsets[e+1] = padded_offsets[e] + ceil(counts[e] / BLOCK_M) * BLOCK_M`.
Launch grid uses `M_padded = padded_offsets[E]`. Each tile now cleanly
belongs to exactly one expert. Store mask ensures no OOB writes.

### Option ii — sorted_token_ids remap (like vLLM's fused_moe.py)

Build a Python-side array `sorted_token_ids [M_padded]` where each entry
is either a valid row index into `sorted_x` OR a sentinel `-1` for padding.
Kernel looks up its rows via this array. More flexible for skewed loads
but more Python-side setup.

**Recommendation**: start with Option (i) padding. Simpler to reason about,
handles skew correctly (empty experts contribute 0 padded rows). Consider
option (ii) only if benchmarking shows padding waste on your input distribution.

## Empty-expert handling

If `offsets[e] == offsets[e+1]`, expert `e` receives zero records. With
Option (i) padding, that expert contributes 0 tiles to the grid — automatic
skip. If you use unpadded offsets + binary search, add an explicit check
in the kernel: `if offsets[e] == offsets[e+1]: return`.

## Correctness invariants (what the tests check)

Test file: `bootcamp/tests/test_ex09_fused_moe.py`.

1. **Small config** (H=128, I=64, E=8, top_k=2, N=64): kernel output equals
   `fused_moe_reference` (Ex05b's PyTorch loop) at fp32 and bf16, uniform
   and skewed routing.

2. **Qwen3-30B-A3B config** (H=2048, I=768, E=128, top_k=8, N=1024):
   larger accumulation → tolerance scaled up. Same fp32/bf16 × uniform/skew
   matrix.

3. **Skewed routing** activates the empty-expert path (half the experts
   receive zero records). Kernel must not divide by zero, launch zero-work
   tiles, or corrupt other experts' output.

Tolerance:
- fp32: `atol=rtol=1e-5 × base`  (base ∈ {4, 16})
- bf16: `atol=rtol=5e-2 × base`  (bf16 has ~7 bit mantissa; a couple % over
  a K=2048 dot product is normal)

## Common pitfalls to watch for

1. **Accumulator dtype.** For bf16 inputs, the matmul accumulator MUST be
   fp32. Explicit `acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)`. Cast
   to output dtype only at store. If small-case passes but Qwen3-scale fails
   at tolerance, this is the culprit 90% of the time.

2. **Weight transpose.** The math is `x @ W.T` (rows of W are output
   channels). In Triton, either load W with transposed strides or use
   `tl.dot(a_tile, tl.trans(b_tile))`. Pick one and stay consistent.

3. **Masking on the K reduction.** When K doesn't divide BLOCK_K, the tail
   k-tile has fewer valid elements. Mask the load. Don't just rely on
   zero-init garbage — some HBM pages hold NaN.

4. **Stride 0 loads on empty experts.** If you keep unpadded offsets and
   forget the empty-expert check, the kernel may load past the tensor.
   Use Option (i) padding OR early-return on `offsets[e] == offsets[e+1]`.

5. **BLOCK_K vs H alignment.** Qwen3 has H=2048 and I=768. Pick BLOCK_K
   that divides both cleanly (32 or 64 works). Non-dividing BLOCK_K is
   handled by the mask but wastes tail-iteration compute.

## Weight-packing helper

`bootcamp/ex09_fused_moe/reference.py::pack_expert_weights` converts an
`nn.ModuleList` of `RefSwiGLU_MLP` (as used in Ex05b/Ex06/Ex07) into the
packed `[E, I, H]` and `[E, H, I]` tensors the kernel wants. This is what
you'll use in Scope S+ / M to plug the kernel into Ex06 / Ex07.

## Progression suggestion

**Day 1** (4-6 h):
- Set up file scaffolding, wire the wrapper + kernel signature.
- Get grouped_matmul_kernel working on **fp32, small config, uniform** —
  no autotune, hand-picked BLOCK_M=32, BLOCK_N=32, BLOCK_K=32.
- Add the Python-side padding helper (Option i).
- One test passing: `test_small[fp32-uniform]`.

**Day 2** (4-6 h):
- Extend to **bf16** — add fp32 accumulator, verify tolerance holds.
- Extend to **skewed routing** — activate empty-expert path.
- Extend to **Qwen3-scale** — tune BLOCK_M/N/K if perf is off.
- All tests in `test_ex09_fused_moe.py` green.

**Day 3-4** (6-8 h):
- Refactor to Option C — fuse gate+up. Update `fused_moe_forward` wrapper.
- Same test suite still passes.
- Intrinsic benchmark: kernel vs `fused_moe_reference` at Qwen3 dims.

**Day 5** (paper-facing):
- Writeup in `learning_summary.md` — bugs hit, tiling decisions, autotune
  configs.
- Save benchmark JSONL + stdout table.

## Scope S+ / M preview (next phases)

After Scope S:

- **Scope S+**: swap Ex06's Phase 7 (Python expert loop) with the kernel.
  Re-run Ex06 lean-vs-dispatch benchmark. Predicted: schedule-choice
  speedup amplifies from 1.09-1.26× → ~1.8-2.5× at ep=8 single-node,
  1.64-4.07× → ~2.5-6× at ep=16 cross-node.
- **Scope M**: swap Ex07's Phase 7. Full HybridBlock end-to-end benchmark
  (including attention TP AR). Paper's Table B.

## Reference implementations available

One consolidated reference sits alongside `solution.py`:

- **`reference.py`** — three things in one file:
  1. `fused_moe_reference` — pure-PyTorch per-expert loop, the
     correctness oracle used by the test suite.
  2. `prepare_sorted_input` / `pack_expert_weights` — test scaffolding
     helpers that build `(sorted_x, offsets)` and pack weights into
     `[E, I, H]` / `[E, H, I]`.
  3. A WORKING Approach 2 (vLLM-style dispatch-table) Triton kernel:
     `fused_moe_forward` + `grouped_matmul_kernel_v2`. All 8 tests
     pass against this. Uses `input_precision="ieee"` on `tl.dot` (not
     TF32) and applies SiLU × mul outside the kernel.

You're expected to write your own kernel in `solution.py`. The Triton
portion of `reference.py` is a known-good target, not something to
copy — a solution that copies it verbatim defeats the exercise. It's
here so you can (a) verify the test suite works, (b) sanity-check
design decisions (grid layout, mask handling, accumulator dtype)
against a working baseline when you get stuck.

## External reference material

- **vLLM `fused_moe.py`** — [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) —
  the production reference. `sorted_token_ids`, `expert_ids`,
  `num_tokens_post_padded` naming.
- **Megablocks Gale et al. 2022** — [arxiv 2211.15841](https://arxiv.org/abs/2211.15841) —
  the "grouped-GEMM formulation of MoE" paper. Same permutation
  vocabulary; Option B (fully-fused) reference.
- **Your CS336 FA-2 backward kernel** — same tiling patterns, minus
  atomics, minus softmax. Grouped-GEMM forward is *easier* than FA-2
  backward. Reuse your `@triton.autotune` config style.
- **Ex05b `bootcamp/ex05_moe_baseline/solution_b.py`** — the PyTorch
  reference this kernel replaces. Same math, different implementation.

## Verifying the test suite is correct

If you want to confirm the test file works before writing anything:

```sh
CUDA_VISIBLE_DEVICES=<n> uv run python -c "
from bootcamp.ex09_fused_moe import reference, solution
solution.fused_moe_forward = reference.fused_moe_forward
import pytest
pytest.main(['-x', '-v', 'bootcamp/tests/test_ex09_fused_moe.py'])
"
```

This swaps `solution.fused_moe_forward` with the working reference and runs
the full suite (~20 seconds on 1×A100). All 8 tests should pass. Once
verified, revert to writing your own kernel in `solution.py`.
