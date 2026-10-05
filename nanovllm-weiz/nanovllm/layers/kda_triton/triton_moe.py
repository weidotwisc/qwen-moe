"""Triton fused MoE for Qwen3-30B-A3B (bf16, single rank / local experts).

Pipeline (vLLM-style sorted grouped GEMM):
  1. route()  -> topk_idx[T,K], topk_weight[T,K]        (torch)
  2. build a per-local-expert sorted+padded row layout    (torch, no host sync)
  3. kernel1: grouped  x @ W13.T  with fused SiLU-gate*up -> inter[Npad, I]  (Triton)
  4. kernel2: grouped  inter @ W2.T  * routing_weight  -> atomic scatter-add out[T,H] (Triton)

Rows are (token, local-expert) assignments. gate = first half of W13 output (HF-native order).
Output is an fp32 partial-sum accumulator cast to the requested dtype.
"""
import torch
import triton
import triton.language as tl

from .config import (
    HIDDEN_SIZE, MOE_INTERMEDIATE, GEMM1_OUT, LOCAL_EXPERTS, TOP_K, DTYPE,
)
from .reference import route


# ----------------------------------------------------------------------------
# Kernel 1: fused gate/up GEMM + SiLU.   inter[pos, :I] = silu(gate) * up
# ----------------------------------------------------------------------------
@triton.jit
def _moe_gemm1_silu_kernel(
    hidden_ptr, w13_ptr, inter_ptr,
    sorted_token_ptr, sorted_valid_ptr, block_expert_ptr,
    H, I,
    stride_hm, stride_hk,
    stride_we, stride_wn, stride_wk,
    stride_im, stride_in,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N
    pid_n = pid % NUM_N

    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:                       # sentinel: unused padding block
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # over I (gate/up column)
    offs_k = tl.arange(0, BLOCK_K)

    token = tl.load(sorted_token_ptr + offs_m)
    valid = tl.load(sorted_valid_ptr + offs_m) != 0

    a_base = hidden_ptr + token[:, None] * stride_hm
    wg_base = w13_ptr + e * stride_we + offs_n[None, :] * stride_wn          # gate rows [0:I)
    wu_base = w13_ptr + e * stride_we + (offs_n[None, :] + I) * stride_wn    # up   rows [I:2I)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        kmask = k < H
        a = tl.load(a_base + k[None, :] * stride_hk,
                    mask=valid[:, None] & kmask[None, :], other=0.0)
        wg = tl.load(wg_base + k[:, None] * stride_wk, mask=kmask[:, None], other=0.0)
        wu = tl.load(wu_base + k[:, None] * stride_wk, mask=kmask[:, None], other=0.0)
        acc_g += tl.dot(a, wg)
        acc_u += tl.dot(a, wu)

    silu = acc_g * tl.sigmoid(acc_g)
    out = (silu * acc_u).to(inter_ptr.dtype.element_ty)
    inter_ptrs = inter_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    tl.store(inter_ptrs, out, mask=valid[:, None])


# ----------------------------------------------------------------------------
# Kernel 2: down GEMM + routing-weight scale + atomic scatter-add into out[T,H].
# ----------------------------------------------------------------------------
@triton.jit
def _moe_gemm2_scatter_kernel(
    inter_ptr, w2_ptr, out_ptr,
    sorted_token_ptr, sorted_valid_ptr, sorted_weight_ptr, block_expert_ptr,
    H, I,
    stride_im, stride_ik,
    stride_we, stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N
    pid_n = pid % NUM_N

    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # over H
    offs_k = tl.arange(0, BLOCK_K)

    token = tl.load(sorted_token_ptr + offs_m)
    valid = tl.load(sorted_valid_ptr + offs_m) != 0
    weight = tl.load(sorted_weight_ptr + offs_m)

    a_base = inter_ptr + offs_m[:, None] * stride_im
    w_base = w2_ptr + e * stride_we + offs_n[None, :] * stride_wn            # rows over H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, I, BLOCK_K):
        k = k0 + offs_k
        kmask = k < I
        a = tl.load(a_base + k[None, :] * stride_ik,
                    mask=valid[:, None] & kmask[None, :], other=0.0)
        w = tl.load(w_base + k[:, None] * stride_wk, mask=kmask[:, None], other=0.0)
        acc += tl.dot(a, w)

    acc = acc * weight[:, None]
    out_ptrs = out_ptr + token[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(out_ptrs, acc, mask=valid[:, None])


# ----------------------------------------------------------------------------
# Host-side layout builder (vectorized torch, fixed-cap -> no device->host sync).
# ----------------------------------------------------------------------------
def num_blocks_for(T, K, num_local, block_m):
    """Fixed worst-case block count (host-known) -> static shapes, CUDA-graph friendly."""
    return (T * K) // block_m + num_local


def _build_sorted_layout(topk_idx, topk_weight, offset, num_local, block_m, device):
    """Static-shape, sync-free layout builder.

    Operates on all T*K (token,slot) rows: non-local rows are tagged with sentinel expert
    `num_local` (sorted to the tail) and routed to a trash slot, so there is no data-dependent
    boolean compaction or host branch. Every tensor has a shape known from (T,K) on the host,
    which makes the whole forward pass capturable by a CUDA graph.
    """
    T, K = topk_idx.shape
    flat_e = topk_idx.reshape(-1)
    flat_tok = torch.arange(T, device=device).repeat_interleave(K).to(torch.int32)
    flat_w = topk_weight.reshape(-1).to(torch.float32)

    local_id = flat_e - offset
    is_local = (local_id >= 0) & (local_id < num_local)
    key = torch.where(is_local, local_id, torch.full_like(local_id, num_local)).to(torch.int32)
    w = torch.where(is_local, flat_w, torch.zeros_like(flat_w))

    num_blocks = num_blocks_for(T, K, num_local, block_m)
    Npad = num_blocks * block_m

    order = torch.argsort(key)                       # local rows first (keys 0..LE-1), tail = LE
    se = key[order]
    st = flat_tok[order]
    sw = w[order]

    # counts per expert via scatter_add (bincount would sync host<->device -> breaks graph capture)
    counts = torch.zeros(num_local + 1, dtype=torch.int64, device=device)
    counts.scatter_add_(0, se.long(), torch.ones(T * K, dtype=torch.int64, device=device))
    counts = counts[:num_local]                                            # [LE]
    blocks_per_e = (counts + block_m - 1) // block_m
    cumsum_blocks = torch.cumsum(blocks_per_e, 0)                           # inclusive block ends
    e_pad_start = (cumsum_blocks - blocks_per_e) * block_m
    cnt_excl = torch.cumsum(counts, 0) - counts

    p = torch.arange(T * K, device=device)
    se_c = se.clamp(max=num_local - 1)
    valid_row = se < num_local
    dest = torch.where(valid_row, e_pad_start[se_c] + (p - cnt_excl[se_c]),
                       torch.full_like(p, Npad))                            # non-local -> trash

    sorted_token = torch.zeros(Npad + 1, dtype=torch.int32, device=device)
    sorted_valid = torch.zeros(Npad + 1, dtype=torch.int32, device=device)
    sorted_weight = torch.zeros(Npad + 1, dtype=torch.float32, device=device)
    sorted_token.scatter_(0, dest, st)
    sorted_valid.scatter_(0, dest, valid_row.to(torch.int32))
    sorted_weight.scatter_(0, dest, sw)

    blk_ids = torch.arange(num_blocks, device=device)
    total_used = cumsum_blocks[-1]
    be = torch.searchsorted(cumsum_blocks, blk_ids, right=True).to(torch.int32)
    block_expert = torch.where(blk_ids < total_used, be, torch.full_like(be, -1))

    return sorted_token[:Npad], sorted_valid[:Npad], sorted_weight[:Npad], block_expert, num_blocks


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
_DEFAULT_CFG = dict(BLOCK_M=64, BLOCK_N1=128, BLOCK_N2=128, BLOCK_K=64,
                    num_warps=4, num_stages=3)


@torch.no_grad()
def moe_forward(hidden_states, w13, w2, routing_logits, local_expert_offset,
                top_k=TOP_K, cfg=None, out_dtype=DTYPE):
    """Local partial MoE output [T, H] (dtype=out_dtype)."""
    cfg = {**_DEFAULT_CFG, **(cfg or {})}
    device = hidden_states.device
    T = hidden_states.shape[0]
    H = HIDDEN_SIZE
    I = MOE_INTERMEDIATE
    LE = w13.shape[0]
    BM = cfg["BLOCK_M"]

    topk_idx, topk_weight = route(routing_logits, top_k)
    sorted_token, sorted_valid, sorted_weight, block_expert, num_blocks = _build_sorted_layout(
        topk_idx, topk_weight, local_expert_offset, LE, BM, device)

    Npad = num_blocks * BM
    inter = torch.empty(Npad, I, dtype=hidden_states.dtype, device=device)
    out_f32 = torch.zeros(T, H, dtype=torch.float32, device=device)

    num_n1 = triton.cdiv(I, cfg["BLOCK_N1"])
    grid1 = (num_blocks * num_n1,)
    _moe_gemm1_silu_kernel[grid1](
        hidden_states, w13, inter,
        sorted_token, sorted_valid, block_expert,
        H, I,
        hidden_states.stride(0), hidden_states.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        inter.stride(0), inter.stride(1),
        BLOCK_M=BM, BLOCK_N=cfg["BLOCK_N1"], BLOCK_K=cfg["BLOCK_K"],
        NUM_N=num_n1, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )

    num_n2 = triton.cdiv(H, cfg["BLOCK_N2"])
    grid2 = (num_blocks * num_n2,)
    _moe_gemm2_scatter_kernel[grid2](
        inter, w2, out_f32,
        sorted_token, sorted_valid, sorted_weight, block_expert,
        H, I,
        inter.stride(0), inter.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out_f32.stride(0), out_f32.stride(1),
        BLOCK_M=BM, BLOCK_N=cfg["BLOCK_N2"], BLOCK_K=cfg["BLOCK_K"],
        NUM_N=num_n2, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out_f32.to(out_dtype)


class MoEGraphRunner:
    """CUDA-graph wrapper over `moe_forward` (candidate C3).

    The whole pipeline (routing + sync-free static layout + both Triton GEMMs) is captured per
    seq_len, collapsing ~1 ms of per-call CPU op-dispatch into a single graph replay. Weights are
    captured by reference; hidden_states / routing_logits are copied into static input buffers
    before each replay. Returns the static output buffer (clone it if retained across replays).
    """

    def __init__(self, w13, w2, local_expert_offset, cfg=None, out_dtype=DTYPE):
        self.w13 = w13
        self.w2 = w2
        self.offset = local_expert_offset
        self.cfg = {**_DEFAULT_CFG, **(cfg or {})}
        self.out_dtype = out_dtype
        self.device = w13.device
        self._graphs = {}

    def _capture(self, T):
        H = HIDDEN_SIZE
        E = self.w13.shape[0]  # unused; kept for clarity
        dev = self.device
        hbuf = torch.zeros(T, HIDDEN_SIZE, dtype=self.w13.dtype, device=dev)
        lbuf = torch.zeros(T, 128, dtype=self.w13.dtype, device=dev)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                moe_forward(hbuf, self.w13, self.w2, lbuf, self.offset,
                            top_k=TOP_K, cfg=self.cfg, out_dtype=self.out_dtype)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = moe_forward(hbuf, self.w13, self.w2, lbuf, self.offset,
                              top_k=TOP_K, cfg=self.cfg, out_dtype=self.out_dtype)
        self._graphs[T] = (g, hbuf, lbuf, out)

    @torch.no_grad()
    def run(self, hidden_states, routing_logits):
        T = hidden_states.shape[0]
        if T not in self._graphs:
            self._capture(T)
        g, hbuf, lbuf, out = self._graphs[T]
        hbuf.copy_(hidden_states)
        lbuf.copy_(routing_logits)
        g.replay()
        return out
