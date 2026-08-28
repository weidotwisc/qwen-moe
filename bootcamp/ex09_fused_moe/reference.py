"""Ex09 reference — Triton grouped-GEMM MoE + PyTorch oracle + test helpers.

Consolidates three things:

1. **PyTorch oracle** (`fused_moe_reference`) — per-expert SwiGLU in pure
   PyTorch. Test correctness oracle for the Triton kernel below.

2. **Test scaffolding** (`prepare_sorted_input`, `pack_expert_weights`) —
   builds the (sorted_x, offsets) tuple that the kernel consumes, plus
   packs an `nn.ModuleList of RefSwiGLU_MLP` into contiguous `[E, I, H]`
   / `[E, H, I]` weight tensors. Used by the tests and any downstream
   integration.

3. **Triton kernel** (`fused_moe_forward` + `grouped_matmul_kernel_v2`) —
   the vLLM-style dispatch-table grouped-GEMM kernel that Ex09 targets.

## Invariants on `fused_moe_forward(...)` (kernel pre/post conditions)

The kernel is delicate — it produces wrong output silently if any of
these invariants are violated. Callers must guarantee them.

### `sorted_x: torch.Tensor`

  **Shape**: `[M, H]` where `M = sorted_x.shape[0]` is the number of
  records this kernel processes. `M` is determined entirely by the
  caller's routing/dispatch context:
    - Standalone Ex09 (tests): `M == N × top_k`, all records local.
    - Integrated with Ex06 (pure EP): `M` is this rank's post-dispatch
      count, which varies by rank and routing distribution. Not
      derivable from N × top_k on the receiving side.
    - Integrated with Ex07 (TP × EP): same as Ex06, further influenced
      by TP striping.

  The kernel is agnostic to which context; it just processes `M`
  records total. `M == offsets[E]` — see the `offsets` invariant below.

  **Dtype**: `torch.float32` or `torch.bfloat16`. Must match all
  `W_*` weight tensor dtypes.
  **Device**: CUDA. Contiguous, row-major (`stride == (H, 1)`).
  **Layout invariant**: rows are sorted by destination expert id.
  Specifically, for every row `i` in `[0, M)`, there exists exactly
  one expert `e ∈ [0, E)` such that `offsets[e] <= i < offsets[e+1]`,
  and row `i`'s content is the hidden state of a token routed to
  expert `e`.

### `offsets: torch.Tensor`

  **Shape**: `[E + 1]`.
  **Dtype**: `torch.int64` (required — the kernel casts to `tl.int32`
  internally, but the Python-side setup relies on int64 arithmetic).
  **Device**: same as sorted_x.
  **Invariants**:
    - `offsets[0] == 0`.
    - `offsets[E] == M == sorted_x.shape[0]`.
    - `offsets[e+1] >= offsets[e]` for every `e ∈ [0, E)` (monotonic
      non-decreasing; equality means expert e is empty).
    - `offsets[e+1] - offsets[e]` = number of records routed to
      expert e.
    - `M` and the per-expert counts are ROUTING-DEPENDENT and
      caller-owned. The kernel never validates them — mismatch
      between `sorted_x.shape[0]` and `offsets[E]` produces silent
      OOB reads.

### `W_gate, W_up: torch.Tensor`

  **Shape**: `[E, I, H]` — PyTorch `[out, in]` layout for each expert's
  gate/up projection matrix.
  **Dtype**: matches sorted_x.
  **Device**: same as sorted_x.

### `W_down: torch.Tensor`

  **Shape**: `[E, H, I]` — PyTorch `[out, in]` layout for each expert's
  down projection matrix.
  **Dtype**: matches sorted_x.

### Postcondition on `fused_moe_forward(...)`

  Returns `out: torch.Tensor` of shape `[M, H]` (dtype and device
  matching sorted_x), such that for every row `i ∈ [0, M)`, letting
  `e` be the unique expert with `offsets[e] <= i < offsets[e+1]`:
  ```
  gate_i   = sorted_x[i] @ W_gate[e].T                     # [I]
  up_i     = sorted_x[i] @ W_up[e].T                       # [I]
  hid_i    = silu(gate_i) * up_i                            # [I]
  out[i]   = hid_i @ W_down[e].T                            # [H]
  ```
  Numerically equal to `fused_moe_reference(...)` up to floating-point
  reduction-order tolerance (fp32: ~1e-5; bf16: ~5e-2 with base scaling
  as in `bootcamp/tests/conftest.py::tol`).

### Non-invariants (deliberately weakened)

  - Empty experts (`offsets[e+1] == offsets[e]`) are allowed and
    correctly skipped.
  - `sorted_x` need not have any particular alignment beyond row-major
    contiguity — `boundary_check + padding_option="zero"` handles the
    tail of each expert's block.
  - `sorted_x` and each `W_*` can be independently allocated (no
    aliasing required).

## Kernel design summary (vLLM-style Approach 2 dispatch table)

Instead of a padded uniform grid (`grid = (E * max_tiles_per_expert, N_tiles)`
with early-return on empty slots), we build a compact dispatch table
in Python and launch `grid = (total_tiles, N_tiles)`. The kernel body
has no branches — every program has real work to do because the
Python setup guarantees it.

The dispatch table has two arrays:
- `tile_expert [total_tiles]`: for each program, which expert.
- `tile_row_start [total_tiles]`: for each program, which row of
  sorted_x its tile begins at.

Both are computed with GPU-vectorized ops (repeat_interleave, cumsum,
arange, gather). ~1-2 μs at Qwen3 scale — negligible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


# =============================================================================
# PyTorch oracle (for correctness comparison in tests)
# =============================================================================


def fused_moe_reference(
    sorted_x: torch.Tensor,     # [M, H]   — M = caller-determined record count; equals offsets[E]
    offsets: torch.Tensor,       # [E + 1], int64
    W_gate: torch.Tensor,        # [E, I, H]
    W_up: torch.Tensor,          # [E, I, H]
    W_down: torch.Tensor,        # [E, H, I]
) -> torch.Tensor:
    """Per-expert SwiGLU on pre-sorted records — the PyTorch correctness oracle.

    For each expert e in [0, E):
        x_chunk = sorted_x[offsets[e] : offsets[e+1]]     # [n_e, H]
        gate    = x_chunk @ W_gate[e].T                    # [n_e, I]
        up      = x_chunk @ W_up[e].T                      # [n_e, I]
        hid     = SiLU(gate) * up                          # [n_e, I]
        out     = hid @ W_down[e].T                        # [n_e, H]
        sorted_out[offsets[e] : offsets[e+1]] = out

    Empty experts (offsets[e] == offsets[e+1]) are skipped — no zero-size
    matmul launches.
    """
    M, H = sorted_x.shape
    E = W_gate.shape[0]
    assert offsets.shape == (E + 1,)
    assert offsets.dtype == torch.int64
    assert W_gate.shape == (E, W_gate.shape[1], H)
    I = W_gate.shape[1]
    assert W_up.shape == (E, I, H)
    assert W_down.shape == (E, H, I)

    out = torch.empty_like(sorted_x)
    for e in range(E):
        s = int(offsets[e].item())
        f = int(offsets[e + 1].item())
        if s == f:
            continue
        x_chunk = sorted_x[s:f]                            # [n_e, H]
        gate = F.linear(x_chunk, W_gate[e])                # [n_e, I]
        up = F.linear(x_chunk, W_up[e])                    # [n_e, I]
        hid = F.silu(gate) * up                            # [n_e, I]
        out[s:f] = F.linear(hid, W_down[e])                # [n_e, H]
    return out


# =============================================================================
# Test scaffolding helpers
# =============================================================================


def prepare_sorted_input(
    x: torch.Tensor,             # [N, H]
    top_k: int,
    num_experts: int,
    router_gate: nn.Linear,       # for producing top-k routing decisions
    skew: bool = False,
) -> dict:
    """Build the (sorted_x, offsets, sorted_token_ids, sorted_weights) tuple
    that Ex09's kernel consumes.

    Under `skew=True`, half the experts get zero records (adversarial load
    imbalance). Used to test the empty-expert code path.

    Returns a dict with:
        sorted_x         [Nk, H]
        offsets           [num_experts + 1]  int64
        sorted_token_ids  [Nk]  int64   — original token position for each record
        sorted_weights    [Nk]           — routing weight per record
        top_k_expert_ids  [N, top_k]  int64  — for reference recomputation
        top_k_weights     [N, top_k]         — for reference recomputation
    """
    device = x.device
    dtype = x.dtype
    N, H = x.shape

    with torch.no_grad():
        logits = router_gate(x)                              # [N, num_experts]
        if skew:
            mask = torch.full_like(logits, float("-inf"))
            mask[:, : num_experts // 2] = 0.0
            logits = logits + mask
        top_k_weights_raw, top_k_expert_ids = torch.topk(logits, top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights_raw, dim=-1).to(dtype)

    top_k_expert_ids_flat = top_k_expert_ids.reshape(-1)      # [Nk]
    top_k_weights_flat = top_k_weights.reshape(-1)             # [Nk]
    token_ids = torch.arange(N, device=device).repeat_interleave(top_k)  # [Nk]

    sorted_expert_ids, sort_perm = torch.sort(top_k_expert_ids_flat, stable=True)
    sorted_token_ids = token_ids[sort_perm]
    sorted_weights = top_k_weights_flat[sort_perm]
    sorted_x = x[sorted_token_ids]                             # [Nk, H]

    counts = torch.bincount(sorted_expert_ids, minlength=num_experts)  # [num_experts]
    offsets = F.pad(counts.cumsum(0), (1, 0)).to(torch.int64)  # [num_experts + 1]

    return {
        "sorted_x": sorted_x,
        "offsets": offsets,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "top_k_expert_ids": top_k_expert_ids,
        "top_k_weights": top_k_weights,
    }


def pack_expert_weights(
    experts: nn.ModuleList,   # ModuleList of RefSwiGLU_MLP
) -> dict:
    """Stack per-expert weights into contiguous [E, I, H] and [E, H, I] tensors.

    Given an `nn.ModuleList` of `RefSwiGLU_MLP` (as Ex05b/Ex06/Ex07 use),
    produce packed weight tensors suitable for the Triton kernel.

    Returns a dict with:
        W_gate  [E, I, H]  — stacked gate projections
        W_up    [E, I, H]  — stacked up projections
        W_down  [E, H, I]  — stacked down projections
    """
    W_gate = torch.stack([e.gate_proj.weight for e in experts], dim=0)  # [E, I, H]
    W_up = torch.stack([e.up_proj.weight for e in experts], dim=0)      # [E, I, H]
    W_down = torch.stack([e.down_proj.weight for e in experts], dim=0)  # [E, H, I]
    return {"W_gate": W_gate, "W_up": W_up, "W_down": W_down}


# =============================================================================
# Triton kernel — Approach 2 (vLLM-style dispatch table)
# =============================================================================


@triton.jit
def grouped_matmul_kernel_v2(
    a_ptr,               # [M, K]
    b_ptr,               # [E, N, K]
    c_ptr,               # [M, N]
    offsets_ptr,         # [E + 1] int64
    tile_expert_ptr,     # [total_tiles] int32 — which expert does this program own
    tile_row_start_ptr,  # [total_tiles] int32 — which row of a/c does this tile begin at
    M, N, K,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped GEMM (Approach 2): each program owns one (expert, row-tile) pair
    determined by a Python-built dispatch table.

    Grid: (total_tiles, cdiv(N, BLOCK_N))
      pid_m = tl.program_id(0)  → index into dispatch table
      pid_n = tl.program_id(1)  → N-axis tile
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Table lookup — no arithmetic decode, no early-return checks.
    e = tl.load(tile_expert_ptr + pid_m).to(tl.int32)
    m_start = tl.load(tile_row_start_ptr + pid_m).to(tl.int32)

    # Expert's end row (for automatic masking on the last tile of each expert).
    e_end = tl.load(offsets_ptr + e + 1).to(tl.int32)

    n_start = pid_n * BLOCK_N

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(e_end, K),
        strides=(stride_am, stride_ak),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

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


def _build_tile_dispatch(offsets: torch.Tensor, BLOCK_M: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Python-side setup: compute the (tile_expert, tile_row_start) dispatch table.

    Precondition:
      - `offsets: [E + 1] int64`, monotonic non-decreasing, `offsets[0] == 0`.
      - `BLOCK_M > 0`.

    Postcondition:
      Returns `(tile_expert, tile_row_start, total_tiles)` such that for
      every program index `i ∈ [0, total_tiles)`:
        - `e := tile_expert[i]` ∈ `[0, E)`.
        - `m := tile_row_start[i]` is BLOCK_M-aligned within expert e's
          block: `offsets[e] <= m < offsets[e + 1]` AND
          `(m - offsets[e]) % BLOCK_M == 0`.
        - Programs are dense-packed: `total_tiles == sum_e ceil(counts[e] / BLOCK_M)`.
        - Empty experts contribute 0 programs. Non-empty experts each contribute
          `ceil(counts[e] / BLOCK_M)` consecutive programs in expert-id order.

    Both outputs are on `offsets.device`, dtype `int32`.
    """
    device = offsets.device
    E = offsets.shape[0] - 1

    counts = offsets[1:] - offsets[:-1]                              # [E]
    tiles_per_expert = (counts + BLOCK_M - 1) // BLOCK_M              # [E] int64
    total_tiles = int(tiles_per_expert.sum().item())

    if total_tiles == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            0,
        )

    tile_expert = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=device),
        tiles_per_expert.to(torch.int32),
    )  # [total_tiles]

    prefix = F.pad(tiles_per_expert.cumsum(0), (1, 0)).to(torch.int32)  # [E+1]

    tile_within_expert = (
        torch.arange(total_tiles, dtype=torch.int32, device=device)
        - prefix[tile_expert.to(torch.int64)]
    )

    tile_row_start = (
        offsets[tile_expert.to(torch.int64)].to(torch.int32)
        + tile_within_expert * BLOCK_M
    )

    return tile_expert, tile_row_start, total_tiles


def _grouped_matmul_v2(
    a: torch.Tensor,        # [M, K]
    b: torch.Tensor,         # [E, N, K]
    offsets: torch.Tensor,   # [E + 1] int64
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Grouped-GEMM launch: c = grouped_matmul(a, b, offsets)."""
    M, K = a.shape
    E, N, K2 = b.shape
    assert K == K2, f"K mismatch: a.shape[1]={K}, b.shape[2]={K2}"
    assert offsets.shape == (E + 1,)
    assert offsets.dtype == torch.int64

    tile_expert, tile_row_start, total_tiles = _build_tile_dispatch(offsets, BLOCK_M)

    if total_tiles == 0:
        return torch.zeros(M, N, dtype=a.dtype, device=a.device)

    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    grid = (total_tiles, triton.cdiv(N, BLOCK_N))

    grouped_matmul_kernel_v2[grid](
        a, b, c,
        offsets,
        tile_expert, tile_row_start,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def fused_moe_forward(
    sorted_x: torch.Tensor,      # [M, H]   — M = caller-determined record count; see module docstring
    offsets: torch.Tensor,        # [E + 1] int64, with offsets[E] == M
    W_gate: torch.Tensor,         # [E, I, H]
    W_up: torch.Tensor,           # [E, I, H]
    W_down: torch.Tensor,         # [E, H, I]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Approach 2 (vLLM-style): three grouped-GEMM launches, each using a
    Python-built dispatch table.

    See module-level docstring for full pre/post conditions on the tensor
    arguments. Briefly:
        Precondition:  sorted_x rows are partitioned by offsets into
            per-expert blocks; sorted_x.shape[0] == offsets[E] == M.
        Postcondition: for row i owned by expert e (offsets[e] ≤ i < offsets[e+1]),
            out[i] = down(silu(gate(sorted_x[i])) * up(sorted_x[i]))
                    where gate/up/down are the linear layers from
                    W_gate[e], W_up[e], W_down[e].
    """
    gate_out = _grouped_matmul_v2(sorted_x, W_gate, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    up_out = _grouped_matmul_v2(sorted_x, W_up, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    hid = F.silu(gate_out) * up_out
    out = _grouped_matmul_v2(hid, W_down, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    return out
