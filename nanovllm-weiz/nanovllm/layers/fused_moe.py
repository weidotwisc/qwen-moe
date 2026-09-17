"""Fused MoE Triton grouped-GEMM kernel.

[traceability] This is component **C9**, vendored verbatim from
`bootcamp/ex09_fused_moe/solution.py` (`fused_moe_forward` + its grouped-matmul
kernel). The VERIFIED artifact and its tests live there; this is a copy so the
engine is self-contained. If you fix C9 in bootcamp, re-sync this file (or later
replace this module with a direct import of the bootcamp version).

Interface (the "Phase-7 boundary" C5-b produces and C9 consumes):
    fused_moe_forward(sorted_x[M,H], offsets[E+1],
                      W_gate[E,I,H], W_up[E,I,H], W_down[E,H,I]) -> sorted_out[M,H]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def grouped_matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    mid_eid_ptr, mid_inexpert_offset_ptr, expert_offsets_ptr,
    M, N, K, E,
    stride_xm, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    eid = tl.load(mid_eid_ptr + mid).to(tl.int32)
    e_start = tl.load(expert_offsets_ptr + eid).to(tl.int32)
    e_end = tl.load(expert_offsets_ptr + eid + 1).to(tl.int32)
    local_mid = tl.load(mid_inexpert_offset_ptr + mid).to(tl.int32)

    X_block_ptr = tl.make_block_ptr(
        base=x_ptr + e_start * stride_xm,
        shape=((e_end - e_start), K),
        offsets=(local_mid * BLOCK_M, 0),
        strides=(stride_xm, stride_xk),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    W_block_ptr = tl.make_block_ptr(
        base=w_ptr + eid * stride_we,
        shape=(N, K),
        offsets=(nid * BLOCK_N, 0),
        strides=(stride_wn, stride_wk),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    Y_block_ptr = tl.make_block_ptr(
        base=y_ptr + e_start * stride_ym,
        shape=((e_end - e_start), N),
        offsets=(local_mid * BLOCK_M, nid * BLOCK_N),
        strides=(stride_ym, stride_yn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    Y_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    T_k = tl.cdiv(K, BLOCK_K)
    for j in range(T_k):
        X_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero")
        W_block = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option="zero")
        W_block_T = tl.trans(W_block, (1, 0))
        Y_tile = tl.dot(X_block, W_block_T, acc=Y_tile, input_precision="ieee")
        X_block_ptr = X_block_ptr.advance((0, BLOCK_K))
        W_block_ptr = W_block_ptr.advance((0, BLOCK_K))

    tl.store(Y_block_ptr, Y_tile.to(Y_block_ptr.type.element_ty), boundary_check=(0, 1))


def _build_meta_data(offsets: torch.Tensor, BLOCK_M: int = 64):
    expert_num = len(offsets) - 1
    eid_numblocks_mappings = ((offsets[1:] - offsets[:-1]) + BLOCK_M - 1) // BLOCK_M
    mid_eid_mappings = torch.repeat_interleave(
        torch.arange(expert_num, device=offsets.device), repeats=eid_numblocks_mappings)
    total_tiles = len(mid_eid_mappings)
    eid_tile_start = eid_numblocks_mappings.cumsum(dim=0) - eid_numblocks_mappings
    mid_inexpert_offset_mappings = torch.arange(total_tiles, device=offsets.device) \
        - torch.repeat_interleave(input=eid_tile_start, repeats=eid_numblocks_mappings)
    return mid_eid_mappings, mid_inexpert_offset_mappings


def _grouped_matmul(x, w, offsets, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    mid_eid_mappings, mid_inexpert_offset_mappings = _build_meta_data(offsets, BLOCK_M)
    total_tiles = len(mid_eid_mappings)
    M, K = x.shape
    E, N, K1 = w.shape
    assert K == K1
    assert E == len(offsets) - 1
    y = torch.zeros(size=(M, N), device=x.device, dtype=x.dtype)
    grid = (total_tiles, triton.cdiv(N, BLOCK_N))
    grouped_matmul_kernel[grid](
        x_ptr=x, w_ptr=w, y_ptr=y,
        mid_eid_ptr=mid_eid_mappings,
        mid_inexpert_offset_ptr=mid_inexpert_offset_mappings,
        expert_offsets_ptr=offsets,
        M=M, N=N, K=K, E=E,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_we=w.stride(0), stride_wn=w.stride(1), stride_wk=w.stride(2),
        stride_ym=y.stride(0), stride_yn=y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y


def fused_moe_forward(
    sorted_x: torch.Tensor,      # [M, H]
    offsets: torch.Tensor,        # [E + 1]  int64
    W_gate: torch.Tensor,         # [E, I, H]
    W_up: torch.Tensor,           # [E, I, H]
    W_down: torch.Tensor,         # [E, H, I]
    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32,
) -> torch.Tensor:
    gated_x = _grouped_matmul(sorted_x, W_gate, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    up_x = _grouped_matmul(sorted_x, W_up, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    z = F.silu(gated_x) * up_x                                      # silu(gate) * up
    y = _grouped_matmul(z, W_down, offsets, BLOCK_M, BLOCK_K, BLOCK_N)  # BLOCK_K/N swapped for down
    return y
