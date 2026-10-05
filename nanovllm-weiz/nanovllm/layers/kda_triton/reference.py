"""fp32 correctness oracle for Qwen3-30B-A3B MoE (single rank / local experts).

Mirrors the verified HF semantics:
  * routing: topk(logits, K) -> softmax(topk_logits)  (== softmax-all -> topk -> renorm)
  * expert : down( silu(gate(x)) * up(x) ), gate = first half of the fused W13 GEMM output.
Everything accumulates in fp32; output cast to the requested dtype by the caller.
"""
import torch
import torch.nn.functional as F

from .config import TOP_K, MOE_INTERMEDIATE, HIDDEN_SIZE


@torch.no_grad()
def route(routing_logits: torch.Tensor, top_k: int = TOP_K):
    """Return (topk_idx [T,K] int64, topk_weight [T,K] fp32).

    softmax over the selected top-k logits == Qwen's softmax-all -> topk -> renormalize.
    """
    logits = routing_logits.to(torch.float32)
    topk_logits, topk_idx = torch.topk(logits, top_k, dim=-1)
    topk_weight = torch.softmax(topk_logits, dim=-1)
    return topk_idx, topk_weight


@torch.no_grad()
def moe_local_reference(
    hidden_states: torch.Tensor,      # [T, H]
    w13: torch.Tensor,                # [LE, 2I, H]  (gate stacked over up)
    w2: torch.Tensor,                 # [LE, H, I]   (down)
    local_expert_offset: int,
    topk_idx: torch.Tensor,           # [T, K] int64 (global expert ids)
    topk_weight: torch.Tensor,        # [T, K] fp32
) -> torch.Tensor:
    """fp32 partial MoE output [T, H] for the local experts owned by this rank."""
    T = hidden_states.shape[0]
    H = HIDDEN_SIZE
    I = MOE_INTERMEDIATE
    LE = w13.shape[0]
    device = hidden_states.device
    out = torch.zeros(T, H, dtype=torch.float32, device=device)

    x_all = hidden_states.to(torch.float32)
    for le in range(LE):
        e = local_expert_offset + le
        hit = topk_idx == e                       # [T, K]
        if not hit.any():
            continue
        tok, slot = hit.nonzero(as_tuple=True)     # unique tokens (topk ids are distinct per row)
        w = topk_weight[tok, slot]                 # [n]
        x = x_all[tok]                             # [n, H]
        g1 = x @ w13[le].to(torch.float32).t()     # [n, 2I]
        gate, up = g1[:, :I], g1[:, I:]
        act = F.silu(gate) * up                    # [n, I]
        o = (act @ w2[le].to(torch.float32).t()) * w.unsqueeze(1)
        out.index_add_(0, tok, o)
    return out


@torch.no_grad()
def moe_full_reference(hidden_states, w13_full, w2_full, routing_logits, top_k: int = TOP_K):
    """Full (all-experts) fp32 reference — used for distributed all-reduce validation."""
    topk_idx, topk_weight = route(routing_logits, top_k)
    return moe_local_reference(hidden_states, w13_full, w2_full, 0, topk_idx, topk_weight)
