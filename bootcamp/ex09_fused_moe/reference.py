"""Reference implementation for Ex09 — pure PyTorch per-expert compute.

Same math as Ex05b's step 6 (per-expert SwiGLU on sorted-by-expert records),
packaged as a standalone function so the Triton kernel in `solution.py` can be
compared against it directly.

Also provides `prepare_sorted_input` — builds the (sorted_x, offsets,
sorted_token_ids, sorted_weights, packed weights) tuple that the kernel
consumes, from an unsorted [N, H] batch + random router. Used by tests
and the microbenchmark.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


def fused_moe_reference(
    sorted_x: torch.Tensor,     # [M, H]   — M = caller-determined record count; equals offsets[E]
    offsets: torch.Tensor,       # [E + 1], int64
    W_gate: torch.Tensor,        # [E, I, H]
    W_up: torch.Tensor,          # [E, I, H]
    W_down: torch.Tensor,        # [E, H, I]
) -> torch.Tensor:
    """Per-expert SwiGLU on pre-sorted records.

    For each expert e in [0, E):
        x_chunk = sorted_x[offsets[e] : offsets[e+1]]     # [n_e, H]
        gate    = x_chunk @ W_gate[e].T                    # [n_e, I]
        up      = x_chunk @ W_up[e].T                      # [n_e, I]
        hid     = SiLU(gate) * up                          # [n_e, I]
        out     = hid @ W_down[e].T                        # [n_e, H]
        sorted_out[offsets[e] : offsets[e+1]] = out

    Empty experts (offsets[e] == offsets[e+1]) are skipped — no zero-size
    matmul launches.

    Args:
        sorted_x: post-dispatch, sorted-by-expert record hidden states.
        offsets: per-expert row boundaries in sorted_x. offsets[e]..offsets[e+1]
            delimits expert e's rows.
        W_gate, W_up, W_down: per-expert weights, packed on axis 0.

    Returns:
        sorted_out: same shape as sorted_x, per-expert SwiGLU applied.
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


def prepare_sorted_input(
    x: torch.Tensor,             # [N, H]
    top_k: int,
    num_experts: int,
    router_gate: nn.Linear,       # for producing top-k routing decisions
    skew: bool = False,
) -> dict:
    """Build the (sorted_x, offsets, sorted_token_ids, sorted_weights) tuple
    that Ex09's kernel consumes, plus per-expert packed weights.

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

    # Router
    with torch.no_grad():
        logits = router_gate(x)                              # [N, num_experts]
        if skew:
            # Adversarial mask: force topk to pick from the first half of experts only.
            mask = torch.full_like(logits, float("-inf"))
            mask[:, : num_experts // 2] = 0.0
            logits = logits + mask
        top_k_weights_raw, top_k_expert_ids = torch.topk(logits, top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights_raw, dim=-1).to(dtype)

    # Flatten
    top_k_expert_ids_flat = top_k_expert_ids.reshape(-1)      # [Nk]
    top_k_weights_flat = top_k_weights.reshape(-1)             # [Nk]
    token_ids = torch.arange(N, device=device).repeat_interleave(top_k)  # [Nk]

    # Sort by expert id
    sorted_expert_ids, sort_perm = torch.sort(top_k_expert_ids_flat, stable=True)
    sorted_token_ids = token_ids[sort_perm]
    sorted_weights = top_k_weights_flat[sort_perm]
    sorted_x = x[sorted_token_ids]                             # [Nk, H]

    # Offsets from bincount
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
