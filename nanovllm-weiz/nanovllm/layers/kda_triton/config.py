"""Verified architecture constants for Qwen3-30B-A3B MoE.

All values confirmed against the actual model config, `transformers` modeling code, and the
checkpoint `model.safetensors.index.json` (see docs/draft.md section 0).
"""
import torch

# --- Model (verified from config.json) ---
HIDDEN_SIZE = 2048          # hidden_size
NUM_EXPERTS = 128           # num_experts (routed)
TOP_K = 8                   # num_experts_per_tok
MOE_INTERMEDIATE = 768      # moe_intermediate_size (I)
GEMM1_OUT = 2 * MOE_INTERMEDIATE  # W13 out (gate+up) = 1536
NORM_TOPK_PROB = True       # renormalize top-k weights
HIDDEN_ACT = "silu"         # SwiGLU: down(silu(gate(x)) * up(x))
NUM_HIDDEN_LAYERS = 48
HAS_SHARED_EXPERTS = False  # verified: no *shared_expert* keys in checkpoint
HAS_ROUTING_BIAS = False    # router is a plain Linear, no bias

# --- Distributed (task contract) ---
EP = 8
LOCAL_EXPERTS = NUM_EXPERTS // EP   # 16 local experts per rank

# --- dtypes ---
DTYPE = torch.bfloat16      # requested activation/weight/output dtype
ACC_DTYPE = torch.float32   # tensor-core accumulation / reference accumulation

# Representative token counts to sweep (decode -> prefill).
SEQ_LENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def local_expert_offset(rank: int) -> int:
    return LOCAL_EXPERTS * rank
