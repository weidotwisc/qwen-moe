"""Exercise 9 — Fused MoE Triton kernel (forward-only).

Port Ex05b's per-expert compute (step 6) to a Triton grouped-GEMM kernel.
Forward only — inference-shape, no autograd, no atomics.

## Interface (matches Ex05b/Ex06's Phase 7 boundary)

Input:
    sorted_x  [Nk, H]              — post-dispatch, sorted by expert id
    offsets   [E_per_rank + 1]     — per-expert row boundaries in sorted_x
    W_gate    [E_per_rank, I, H]   — packed per-expert gate projection
    W_up      [E_per_rank, I, H]   — packed per-expert up projection
    W_down    [E_per_rank, H, I]   — packed per-expert down projection

Output:
    sorted_out [Nk, H]             — per-expert SwiGLU applied

Math (per expert e, on rows offsets[e]..offsets[e+1] of sorted_x):
    x_chunk = sorted_x[offsets[e]:offsets[e+1]]           # [n_e, H]
    gate    = x_chunk @ W_gate[e].T                        # [n_e, I]
    up      = x_chunk @ W_up[e].T                          # [n_e, I]
    hid     = silu(gate) * up                              # [n_e, I]
    out     = hid @ W_down[e].T                            # [n_e, H]

Everything is atomics-free — each expert's output block is disjoint.

## Progression

- **Option A (this file's starting point)**: three separate grouped-GEMM
  launches (gate, up, down). Simplest to write; ~2× slower than Option C
  because gate and up each reload sorted_x.
- **Option C (stretch, in the same file)**: fuse gate + up into ONE kernel
  that reads sorted_x once and produces both outputs (paired-column matmul).
  Separate down kernel unchanged. This is what production vLLM does.
- **Option B (research stretch, out of scope for now)**: fully-fused
  gate + up + silu*mul + down in one kernel. See README.md for the
  nested-reduction structure.

## Run

```sh
CUDA_VISIBLE_DEVICES=0 uv run pytest bootcamp/tests/test_ex09_fused_moe.py -v
```
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# =============================================================================
# Option A: Grouped GEMM kernel — parameterized for gate/up/down
# =============================================================================


@triton.jit
def grouped_matmul_kernel(
    # Pointers
    a_ptr,           # [M, K]   input (sorted_x for gate/up; silu_mul_out for down)
    b_ptr,           # [E, N, K]  per-expert weight
    c_ptr,           # [M, N]   output
    offsets_ptr,     # [E + 1]  int64
    # Sizes
    M, N, K, E,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Meta
    MAX_TILES_PER_EXPERT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped GEMM: c[m, n] = sum_k a[m, k] * b[e(m), n, k]

    Recommended grid: **expert-first**.
        grid = (E * MAX_TILES_PER_EXPERT, cdiv(N, BLOCK_N))
    Each program handles one tile of ONE expert — no tile straddles expert
    boundaries. Cleaner than binary-search-then-mask.

    ## What to implement (using block pointers like your CS336 FA-2)

    TODO(you):

    1. **Grid decomposition.**
         pid_full = tl.program_id(0)
         pid_n    = tl.program_id(1)
         e          = pid_full // MAX_TILES_PER_EXPERT
         tile_in_e  = pid_full %  MAX_TILES_PER_EXPERT
       If e >= E: return.

    2. **Expert row range.**
         e_start = tl.load(offsets_ptr + e).to(tl.int32)
         e_end   = tl.load(offsets_ptr + e + 1).to(tl.int32)
         n_e     = e_end - e_start
       If tile_in_e * BLOCK_M >= n_e: return. (This tile-slot is past
       expert e's row count → nothing to do.)

    3. **Tile origin.**
         m_start = e_start + tile_in_e * BLOCK_M
         n_start = pid_n * BLOCK_N

    4. **Block pointer for `a`.** Note `shape=(e_end, K)` — bounding the
       declared shape to the expert's end lets `boundary_check` mask off rows
       past the expert boundary automatically:
         a_bp = tl.make_block_ptr(
             base=a_ptr,
             shape=(e_end, K),
             strides=(stride_am, stride_ak),
             offsets=(m_start, 0),
             block_shape=(BLOCK_M, BLOCK_K),
             order=(1, 0),
         )

    5. **Block pointer for `b`.** Per-expert base via `b_ptr + e * stride_be`:
         b_bp = tl.make_block_ptr(
             base=b_ptr + e * stride_be,
             shape=(N, K),
             strides=(stride_bn, stride_bk),
             offsets=(n_start, 0),
             block_shape=(BLOCK_N, BLOCK_K),
             order=(1, 0),
         )

    6. **Accumulator in fp32.** REQUIRED for bf16 correctness at K=2048.
         acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    7. **Matmul loop over K.**
         for _ in range(0, K, BLOCK_K):
             a_tile = tl.load(a_bp, boundary_check=(0, 1), padding_option="zero")
             b_tile = tl.load(b_bp, boundary_check=(0, 1), padding_option="zero")
             acc = tl.dot(a_tile, tl.trans(b_tile), acc, input_precision="ieee")
             a_bp = tl.advance(a_bp, (0, BLOCK_K))
             b_bp = tl.advance(b_bp, (0, BLOCK_K))

       `input_precision="ieee"` matters — Triton defaults to TF32 on A100
       for fp32 inputs, losing ~1e-3 accuracy vs PyTorch's fp32. Without this
       flag, the fp32 tests fail on tolerance.

    8. **Store output** via a c-block-pointer with the same `shape=(e_end, N)`
       trick:
         c_bp = tl.make_block_ptr(
             base=c_ptr,
             shape=(e_end, N),
             strides=(stride_cm, stride_cn),
             offsets=(m_start, n_start),
             block_shape=(BLOCK_M, BLOCK_N),
             order=(1, 0),
         )
         tl.store(c_bp, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))

    ## Alternative: raw-pointer style

    If you prefer explicit `offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak`
    broadcast arithmetic + manual masks, that also works — vLLM's
    `fused_moe.py` uses it. Same machine code after compilation. Pick
    whichever is more comfortable — block pointers are more concise.

    ## Notes

    - Accumulator MUST be fp32 for bf16 inputs — the accumulator dtype is
      NOT inferred from input dtype.
    - The weight is [N, K] per expert; we do `a @ b.T` via
      `tl.dot(a_tile, tl.trans(b_tile))`.
    - Empty-expert (offsets[e] == offsets[e+1]) → n_e = 0 → step 2's return
      triggers on tile 0 → no work launched. Free.
    """
    raise NotImplementedError("Implement grouped_matmul_kernel — see docstring")


# =============================================================================
# Python wrappers — launch the kernel(s)
# =============================================================================


def _grouped_matmul(
    a: torch.Tensor,              # [M, K]
    b: torch.Tensor,               # [E, N, K]
    offsets: torch.Tensor,         # [E + 1]  int64
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Launch grouped_matmul_kernel for a @ b[e].T per-expert.

    TODO(you):

    1. Extract M, K = a.shape; E, N, K2 = b.shape. Assert K == K2 and
       offsets.shape == (E + 1,) and offsets.dtype == torch.int64.

    2. Compute per-expert row counts and tiles-per-expert:
         counts = (offsets[1:] - offsets[:-1]).to("cpu")   # move to CPU for .max()
         tiles_per_expert = (counts + BLOCK_M - 1) // BLOCK_M
         max_tiles = max(1, int(tiles_per_expert.max().item()))
       The `max(1, ...)` guards against the pathological all-empty case where
       max_tiles=0 would give grid[0]=0 → no kernel launches → uninit output.

    3. Allocate c = torch.empty(M, N, dtype=a.dtype, device=a.device).

    4. grid = (E * max_tiles, triton.cdiv(N, BLOCK_N))

    5. Launch grouped_matmul_kernel[grid](
           a, b, c, offsets,
           M, N, K, E,
           a.stride(0), a.stride(1),
           b.stride(0), b.stride(1), b.stride(2),
           c.stride(0), c.stride(1),
           MAX_TILES_PER_EXPERT=max_tiles,
           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
       )

    6. Return c.
    """
    raise NotImplementedError


def fused_moe_forward(
    sorted_x: torch.Tensor,      # [Nk, H]
    offsets: torch.Tensor,        # [E + 1]  int64
    W_gate: torch.Tensor,         # [E, I, H]
    W_up: torch.Tensor,           # [E, I, H]
    W_down: torch.Tensor,         # [E, H, I]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Option A: three separate grouped-GEMM launches.

    Pipeline:
        gate_out = _grouped_matmul(sorted_x, W_gate, offsets)   # [Nk, I]
        up_out   = _grouped_matmul(sorted_x, W_up,   offsets)   # [Nk, I]
        hid      = F.silu(gate_out) * up_out                     # [Nk, I]
        out      = _grouped_matmul(hid, W_down, offsets)         # [Nk, H]
        return out

    TODO(you): the four lines above, wired up with the kernel wrapper.
    """
    raise NotImplementedError


# =============================================================================
# Option C (stretch goal): fuse gate + up into a single kernel
# =============================================================================
#
# Option C replaces the two separate gate/up launches with ONE kernel that
# reads sorted_x once and produces both gate and up outputs. Downstream
# silu*mul stays elementwise; down stays as its own kernel.
#
# Rough shape:
#   gate_up_out [Nk, 2I]  packed as [gate_slice | up_slice]
#     → one @triton.jit that iterates the N dimension over 2I columns,
#       loading either W_gate or W_up depending on whether n < I
#     → one launch instead of two
#
# TODO(later): implement `grouped_matmul_gate_up_kernel` and
# `fused_moe_forward_optC(...)` that returns the same output as Option A but
# with the fused gate+up launch. Same test suite should pass unchanged.
#
# =============================================================================
