"""Reference Triton implementation for Ex09 — Option A (three grouped-GEMM launches).

Uses **block pointers** (`tl.make_block_ptr` + `tl.advance` + `boundary_check`),
the modern Triton idiom taught in CS336 FA-2. Equivalent to the raw-pointer
style used in vLLM's `fused_moe.py` — both compile to identical machine code.

This is a WORKING reference: `fused_moe_forward` here passes all 8 tests in
`bootcamp/tests/test_ex09_fused_moe.py`. Wei can consult this when stuck on
`solution.py`, and Scope S+/M can call this while Wei's solution is being written.

Design decisions (see README.md for full context):

- **Grid layout**: expert-first. Grid `(E * max_tiles_per_expert, cdiv(N, BLOCK_N))`.
  Each program computes one tile of ONE expert. No tile straddles expert
  boundaries.

- **Expert-bounded row range**: block pointer declared with `shape=(e_end, K)`
  so `boundary_check=(0, 1)` on load automatically masks off rows past the
  expert's end. Elegant handling of the "tile might extend past expert e's
  block" edge case.

- **Empty-expert handling**: expert-first grid naturally skips
  (`m_local >= n_e → return`) — one empty program per unused tile-slot, no
  wasted matmul.

- **Accumulator**: fp32 always, cast to output dtype at store. Required for
  bf16 correctness at Qwen3-30B-A3B dims (K=2048).

- **IEEE fp32 in tl.dot**: `input_precision="ieee"`. Triton defaults to TF32
  on A100 for fp32 inputs, which loses ~1e-3 accuracy vs PyTorch's fp32
  matmul. Explicitly override for correctness.

- **Activation**: applied OUTSIDE the kernel (in `fused_moe_forward` via
  `F.silu(gate) * up`). Keeps the kernel a pure grouped-GEMM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def grouped_matmul_kernel(
    a_ptr,           # [M, K]
    b_ptr,           # [E, N, K]
    c_ptr,           # [M, N]
    offsets_ptr,     # [E + 1] int64
    M, N, K,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    E,
    MAX_TILES_PER_EXPERT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped GEMM using block pointers: c[m, n] = sum_k a[m, k] * b[e(m), n, k]

    Grid: (E * MAX_TILES_PER_EXPERT, cdiv(N, BLOCK_N))
      pid_full = tl.program_id(0)  → (expert_id, tile_within_expert)
      pid_n    = tl.program_id(1)
    """
    pid_full = tl.program_id(0)
    pid_n = tl.program_id(1)

    e = pid_full // MAX_TILES_PER_EXPERT
    tile_in_e = pid_full % MAX_TILES_PER_EXPERT

    if e >= E:
        return

    e_start = tl.load(offsets_ptr + e).to(tl.int32)
    e_end = tl.load(offsets_ptr + e + 1).to(tl.int32)
    n_e = e_end - e_start

    m_local = tile_in_e * BLOCK_M
    if m_local >= n_e:
        return  # This tile-slot is beyond expert e's row count → no work.

    m_start = e_start + m_local
    n_start = pid_n * BLOCK_N

    # Block pointer for `a` — bounded to (e_end, K) so boundary_check
    # automatically masks off rows past expert e's block.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(e_end, K),
        strides=(stride_am, stride_ak),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # Block pointer for `b` — base includes per-expert offset `e * stride_be`.
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + e * stride_be,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(n_start, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a_tile, tl.trans(b_tile), acc, input_precision="ieee")
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(e_end, N),
        strides=(stride_cm, stride_cn),
        offsets=(m_start, n_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def _grouped_matmul(
    a: torch.Tensor,          # [M, K]
    b: torch.Tensor,           # [E, N, K]
    offsets: torch.Tensor,     # [E + 1] int64
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Launch grouped_matmul_kernel to compute c[m, n] = sum_k a[m, k] * b[e(m), n, k]."""
    M, K = a.shape
    E, N, K2 = b.shape
    assert K == K2, f"K mismatch: a.shape[1]={K}, b.shape[2]={K2}"
    assert offsets.shape == (E + 1,)
    assert offsets.dtype == torch.int64

    counts = (offsets[1:] - offsets[:-1]).to("cpu")
    tiles_per_expert = (counts + BLOCK_M - 1) // BLOCK_M
    max_tiles = max(1, int(tiles_per_expert.max().item()))

    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    grid = (E * max_tiles, triton.cdiv(N, BLOCK_N))

    grouped_matmul_kernel[grid](
        a, b, c, offsets,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        E,
        MAX_TILES_PER_EXPERT=max_tiles,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def fused_moe_forward(
    sorted_x: torch.Tensor,     # [M, H]   — M = caller-determined record count
    offsets: torch.Tensor,       # [E + 1] int64, with offsets[E] == M
    W_gate: torch.Tensor,        # [E, I, H]
    W_up: torch.Tensor,          # [E, I, H]
    W_down: torch.Tensor,        # [E, H, I]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Option A: three grouped-GEMM launches + PyTorch elementwise silu×mul.

    `M = sorted_x.shape[0]` is the number of records this kernel
    processes. It equals `offsets[E]` by precondition. In the
    standalone Ex09 test, `M = N × top_k`. In post-dispatch
    integration (Ex06/Ex07), M is this rank's routing-dependent
    received-record count and is unrelated to any single rank's
    (N × top_k).

    Pipeline:
        gate_out = grouped_matmul(sorted_x, W_gate, offsets)   # [M, I]
        up_out   = grouped_matmul(sorted_x, W_up,   offsets)   # [M, I]
        hid      = silu(gate_out) * up_out                      # [M, I]
        out      = grouped_matmul(hid, W_down, offsets)         # [M, H]
    """
    gate_out = _grouped_matmul(sorted_x, W_gate, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    up_out = _grouped_matmul(sorted_x, W_up, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    hid = F.silu(gate_out) * up_out
    out = _grouped_matmul(hid, W_down, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    return out
