"""Qwen3-MoE (e.g. Qwen3-30B-A3B) for nano-vLLM.

From-scratch model adaptation implemented against the HF reference
(transformers/models/qwen3_moe/modeling_qwen3_moe.py). The ONLY architectural
delta from dense Qwen3 is the FFN: dense MLP -> sparse MoE. Everything else
(GQA attention with QK-norm, RoPE, RMSNorm, embeddings) is reused from
`nanovllm.models.qwen3` unchanged.

Scope: single-GPU AND expert-parallel [C6, DP=1 / all_reduce]. Experts are stored
STACKED (w_gate/w_up/w_down) -- the one representation shared by both expert
kernels; `MOE_KERNEL=loop|fused` (env var) selects the compute. Under EP each rank
holds its shard (num_experts/ep_size experts) and combines partials with ONE
all_reduce; ep_size := world_size == tp_size, so TP already replicates the batch
across ranks -- the "DP=1" precondition the all_reduce schedule needs. ep_size==1
is the original single-GPU path, byte-identical. C7 (TP=1/dispatch) + C8 (hybrid) later.

Traceability (for the paper's "generated code -> contracts" thread). Greppable
tags map integration lines back to the verified bootcamp components:
  [C5-b]  bootcamp/ex05_moe_baseline/reference_b.py -- routing: sort -> offsets ->
          grouped compute -> unpermute/combine (and the loop expert path).
  [C9]    bootcamp/ex09_fused_moe/solution.py -- fused grouped-GEMM expert path
          (vendored into nanovllm/layers/fused_moe.py).
  [C6]    bootcamp/ex06_ep/solution_lean.py -- lean expert parallelism (DP=1):
          filter routing records to this rank's LOCAL experts, local grouped
          compute, ONE all_reduce to sum partials. Valid because the MoE input is
          replicated across the EP group (TP provides that when tp==ep==world_size).
  [contract RT1/RT2/RT5]  ex05 routing-partition contracts on `offsets`
          (o[0]=0, o[E]=T*top_k, monotone, block size == true per-expert count).
To reverse the mapping: grep a tag to see which verified component a line realizes
and which invariant a runtime violation there would implicate.
"""
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3MoeConfig

import os

# --- reuse the dense Qwen3 attention unchanged (HF does the same via
#     `class Qwen3MoeAttention(Qwen3Attention)`) ---------------------------------
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
# [C9] fused_moe_forward is imported lazily inside the fused branch below, so the
# default loop path carries no Triton dependency.


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block: router + STACKED experts, with a swappable kernel.

    Experts are stored stacked (`w_gate/w_up/w_down`, shapes matching C9's
    interface) -- the single representation shared by both expert paths:
      MOE_KERNEL=loop  -> [C5-b] per-expert grouped compute on the stack
      MOE_KERNEL=fused -> [C9]   fused Triton grouped-GEMM
    The routing (sort -> offsets -> gather -> unpermute/combine) is [C5-b] either
    way; only the boxed compute step differs. This is the kernel-swappable slot
    that the EP schedules (C6-C8) will later wrap with dispatch/combine.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts            # E = 128 (global)
        self.top_k = config.num_experts_per_tok          # 8
        self.norm_topk_prob = config.norm_topk_prob       # True
        H, I = config.hidden_size, config.moe_intermediate_size   # 2048, 768
        # Which expert kernel to run: "loop" = C5-b reference, "fused" = C9.
        self.kernel = os.environ.get("MOE_KERNEL", "loop")

        # [C6] Expert parallelism, DP=1 variant. ep_size := world_size (== tp_size in
        # nano-vLLM's flat group); nano-vLLM's TP already replicates the batch across
        # ranks, so the MoE input is replicated -- the precondition the all_reduce
        # schedule needs. Each rank owns a contiguous shard of experts_per_rank
        # experts. ep_size==1 -> experts_per_rank==num_experts, expert_start==0:
        # the original single-GPU path. (Guarded so the block is constructible even
        # without an initialized process group, e.g. in a standalone test.)
        if dist.is_initialized():
            self.ep_size, self.ep_rank = dist.get_world_size(), dist.get_rank()
        else:
            self.ep_size, self.ep_rank = 1, 0
        assert config.num_experts % self.ep_size == 0, (
            f"num_experts={config.num_experts} not divisible by ep_size={self.ep_size}")
        self.experts_per_rank = config.num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.experts_per_rank
        self.expert_end = self.expert_start + self.experts_per_rank

        # [C5 router] gate over ALL experts; replicated on every rank (full routing).
        # HF name `...mlp.gate.weight` -> param `self.gate.weight`.
        self.gate = ReplicatedLinear(H, config.num_experts, bias=False)

        # [C9 Phase-7 boundary] experts stored STACKED -- exactly the layout ex09's
        # fused_moe_forward consumes. Under EP only this rank's shard is materialized
        # (experts_per_rank, not the global E). One store, both kernels index it.
        Eloc = self.experts_per_rank
        self.w_gate = nn.Parameter(torch.empty(Eloc, I, H))   # [Eloc, I, H]
        self.w_up   = nn.Parameter(torch.empty(Eloc, I, H))   # [Eloc, I, H]
        self.w_down = nn.Parameter(torch.empty(Eloc, H, I))   # [Eloc, H, I]
        for p in (self.w_gate, self.w_up, self.w_down):
            p.weight_loader = self._expert_weight_loader

    def _expert_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int):
        # HF ships one weight per expert (experts.{e}.{gate,up,down}_proj.weight);
        # utils/loader.py parses the GLOBAL `e` and routes each here.
        # [C6] EP shard: skip experts this rank doesn't own; map global e -> local
        # slice e-expert_start. (loader.py stays generic; the shard logic lives here,
        # mirroring how QKVParallelLinear shards its rows by tp_rank.)
        if not (self.expert_start <= expert_id < self.expert_end):
            return
        param.data[expert_id - self.expert_start].copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ============ [C5-b] routing (ex05_moe_baseline/reference_b.py) ============
        # Router runs over ALL experts on the replicated input -- identical on every
        # rank (global routing decisions). fp32 softmax -> top-k -> renormalize.
        num_tokens = hidden_states.size(0)
        router_logits = self.gate(hidden_states)                                   # [T, E]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # flatten to T*top_k (token, expert, weight) records
        token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(self.top_k)
        expert_ids = selected_experts.reshape(-1)
        weights = routing_weights.reshape(-1)

        # ============ [C6] filter routing records to this rank's LOCAL experts =====
        # Renormalization already happened above over the full top-k, so each kept
        # (token, expert) weight is final. Map global expert id -> local [0, Eloc).
        # ep_size==1 -> no filtering, expert_ids stay global==local (expert_start=0).
        if self.ep_size > 1:
            local_mask = (expert_ids >= self.expert_start) & (expert_ids < self.expert_end)
            token_ids = token_ids[local_mask]
            weights = weights[local_mask]
            expert_ids = expert_ids[local_mask] - self.expert_start                 # -> local ids

        # sort by (local) expert id
        permutation = torch.argsort(expert_ids)
        sorted_expert_ids = expert_ids[permutation]
        sorted_token_ids = token_ids[permutation]
        sorted_weights = weights[permutation]

        # [contract RT1/RT2/RT5] offsets over the LOCAL partition: o[0]=0,
        # o[Eloc]=#local records (== T*top_k only when ep_size==1), monotone,
        # block == per-(local-)expert count.
        counts = torch.bincount(sorted_expert_ids, minlength=self.experts_per_rank)
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=hidden_states.device),
            counts.cumsum(0),
        ])                                                                          # [Eloc + 1]
        sorted_x = hidden_states[sorted_token_ids]                                  # [M, H]

        # ============ [C6] partial output (zeros on tokens with no local expert) ===
        output = torch.zeros_like(hidden_states)                                    # [T, H]
        if sorted_x.size(0) > 0:                        # this rank owns some (token, expert) records
            # ---- expert compute: C5-b loop OR C9 fused, on the LOCAL expert stack ----
            # Same (sorted_x, offsets, stacked W) in -> sorted_out [M, H] out; only
            # this box changes with MOE_KERNEL, the routing/combine around it is shared.
            if self.kernel == "fused":
                # [C9] fused Triton grouped-GEMM (nanovllm/layers/fused_moe.py)
                from nanovllm.layers.fused_moe import fused_moe_forward
                sorted_out = fused_moe_forward(sorted_x, offsets, self.w_gate, self.w_up, self.w_down)
            else:
                # [C5-b] per-expert grouped compute on the stack; SwiGLU, identical math
                # to ex05b/ex09: down( silu(x @ Wg^T) * (x @ Wu^T) ).
                sorted_out = torch.empty_like(sorted_x)
                for e in range(self.experts_per_rank):
                    start, end = int(offsets[e]), int(offsets[e + 1])   # host sync per expert;
                    if start == end:                                    # fused path removes it
                        continue
                    xe = sorted_x[start:end]                            # [n_e, H]
                    g = xe @ self.w_gate[e].t()                         # [n_e, I]
                    u = xe @ self.w_up[e].t()                           # [n_e, I]
                    h = F.silu(g) * u                                   # [n_e, I]
                    sorted_out[start:end] = h @ self.w_down[e].t()      # [n_e, H]

            # ---- [C5-b] weight + unpermute + scatter into the partial ----
            sorted_out = sorted_out * sorted_weights.unsqueeze(-1)
            output.index_add_(0, sorted_token_ids, sorted_out)

        # ============ [C6] EP combine: one all_reduce sums partials across ranks ====
        # Collective -> UNCONDITIONAL when ep_size>1 (every rank must participate,
        # including ranks that owned no tokens this step, or NCCL deadlocks). Uses the
        # default (flat tp==ep) group; each (token,expert) is computed on exactly one
        # rank, so the SUM reconstructs the full output with no double-counting.
        if self.ep_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    """Same as the dense decoder layer, but MLP -> sparse MoE block.

    NOTE: For Qwen3-30B-A3B every layer is MoE (decoder_sparse_step=1,
    mlp_only_layers=[]). If you ever target a model with mixed layers, branch on
    `(layer_idx not in config.mlp_only_layers) and ((layer_idx+1) % config.decoder_sparse_step == 0)`.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MoeSparseMoeBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    # Only attention QKV is fused here. MoE experts are stored STACKED
    # (w_gate/w_up/w_down) and loaded per-expert by utils/loader.py's expert
    # branch, so gate_proj/up_proj are NOT in this mapping (an all-MoE model has
    # no dense MLP). The router `mlp.gate.weight` loads verbatim.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


# --- Registration -------------------------------------------------------------
# nano-vLLM's model_runner.py hardcodes Qwen3ForCausalLM. To load this model,
# dispatch on the architecture, e.g. in engine/model_runner.py:
#
#     arch = hf_config.architectures[0]
#     if arch == "Qwen3MoeForCausalLM":
#         from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM as ModelCls
#     else:
#         from nanovllm.models.qwen3 import Qwen3ForCausalLM as ModelCls
#     self.model = ModelCls(hf_config)
