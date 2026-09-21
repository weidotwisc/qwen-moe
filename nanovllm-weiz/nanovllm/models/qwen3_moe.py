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
is the original single-GPU path, byte-identical.

Scope also includes [C7/C8] via MOE_EP_MODE=hybrid: the general HybridMoE (stripe ->
all-to-all dispatch -> local experts -> combine -> all_gather). Step 1 runs it at
tp_group == ep_group == world (DP=1, no engine change), which validates the dispatch/
stripe/gather collectives; tp_size < world (true C7/C8) awaits the device mesh + engine
DP. See Qwen3MoeSparseMoeBlock._forward_hybrid.

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
  [C7/C8] bootcamp/ex07_tp_ep_hybrid/solution.py -- hybrid TP-scoped stripe/gather +
          EP-scoped all-to-all dispatch/combine (_forward_hybrid). C7 == this block at
          tp_size==1, C8 at tp_size==4; step 1 runs it at tp_size==ep_size==world.
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
        # Which EP schedule to run AROUND that kernel (orthogonal to MOE_KERNEL):
        #   "allreduce" (default) -> [C6] DP=1: replicated input, filter routing to
        #                            local experts, local compute, ONE all_reduce.
        #                            (The validated path; see forward() below.)
        #   "hybrid"              -> [C7/C8] stripe within tp_group -> all-to-all
        #                            DISPATCH across ep_group -> local compute ->
        #                            combine -> all_gather within tp_group.
        #                            (See _forward_hybrid; the ported ex07 block.)
        # Consumed by forward() below. Only meaningful under EP (ep_size>1);
        # ep_size==1 always takes the single-GPU path, byte-identical regardless.
        self.mode = os.environ.get("MOE_EP_MODE", "allreduce")

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

    def _expert_compute(self, sorted_x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """The one swappable compute box, shared by every EP schedule.

        Input `sorted_x` [M, H] is grouped by LOCAL expert per `offsets` [Eloc+1]
        (o[0]=0, o[Eloc]=M, monotone; block e == count for local expert e). Returns
        `sorted_out` [M, H]. Only this box changes with MOE_KERNEL; the routing /
        dispatch / combine around it is identical. Called by BOTH the [C6] all_reduce
        forward and the [C7/C8] _forward_hybrid dispatch forward -- factoring it out is
        exactly why experts are stored STACKED (one representation, both kernels, both
        schedules). Assumes M > 0 (callers guard the empty-partition case).
        """
        if self.kernel == "fused":
            # [C9] fused Triton grouped-GEMM (nanovllm/layers/fused_moe.py)
            from nanovllm.layers.fused_moe import fused_moe_forward
            return fused_moe_forward(sorted_x, offsets, self.w_gate, self.w_up, self.w_down)
        # [C5-b] per-expert grouped compute on the stack; SwiGLU, identical math to
        # ex05b/ex09: down( silu(x @ Wg^T) * (x @ Wu^T) ).
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
        return sorted_out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [C7/C8] Hybrid dispatch schedule (stripe -> all-to-all dispatch -> local
        # experts -> combine -> all_gather), selected by MOE_EP_MODE=hybrid. Only under
        # EP; the default "allreduce" (and every ep_size==1 run) falls through to the
        # validated [C6] path below, byte-identical to before.
        if self.ep_size > 1 and self.mode == "hybrid":
            return self._forward_hybrid(hidden_states)
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
            # ---- expert compute: the shared swappable box (C5-b loop / C9 fused) ----
            sorted_out = self._expert_compute(sorted_x, offsets)
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

    @staticmethod
    def _all_to_all_v(x: torch.Tensor, in_splits: list[int], out_splits: list[int],
                      group=None) -> torch.Tensor:
        """Variable-split all-to-all (the EP dispatch primitive), dependency-free.

        Sends in_splits[r] rows of `x` to rank r and receives out_splits[r] rows from
        rank r; the splits are pre-negotiated by the caller (an all_to_all_single on
        counts). Returns a fresh [sum(out_splits), *x.shape[1:]] tensor. Thin wrapper
        over dist.all_to_all_single so nano carries no bootcamp import. Works for 1-D
        `x` (out_splits sum -> [M]) and 2-D [M, H]; 0-row splits are fine.
        """
        out = x.new_empty((sum(out_splits), *x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(),
                               output_split_sizes=out_splits,
                               input_split_sizes=in_splits, group=group)
        return out

    def _forward_hybrid(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[C7/C8] Hybrid EP forward: TP-scoped stripe/gather + EP-scoped dispatch/combine.

        Ported from bootcamp/ex07_tp_ep_hybrid/solution.py, adapted to nano: experts are
        STACKED (reuse self._expert_compute, not an nn.ModuleList) and routing is nano/HF
        (fp32 softmax over ALL experts -> topk -> renorm, NOT ex07's topk-then-softmax).

        Step-1 wiring: tp_group == ep_group == the flat world, so tp_size == ep_size ==
        world_size and DP = world/tp_size == 1 -- nano's current replicated-batch engine,
        NO engine change. Input is replicated across the tp_group (nano's TP provides
        that); Phase 0 stripes it 1/tp_size so the MoE runs once instead of tp_size-
        redundantly, and Phase 11 all_gathers back to a replicated output. The all-to-all
        DISPATCH is load-bearing (rank r's local experts must receive other ranks' tokens);
        at tp==world it is the *redundant* case the paper contrasts with C6's all_reduce --
        identical result, its job here is to de-risk the collective machinery. (Step 2's
        device mesh replaces the three group locals below so tp_size < world becomes
        expressible: tp=1 -> C7, tp=4 -> C8.)

        Block invariant: {input replicated within tp_group} -> {output replicated within tp_group}.
        """
        # [step 1] tp_group == ep_group == the flat world; DP = world/tp_size == 1.
        # Step 2 (device mesh) replaces these three lines with real subgroups.
        tp_size, tp_rank, tp_group = self.ep_size, self.ep_rank, None
        ep_size, ep_group = self.ep_size, None

        N, H = hidden_states.size(0), hidden_states.size(1)
        dev = hidden_states.device

        # ---- Phase 0: pad N to a multiple of tp_size, then stripe within tp_group ----
        # Every tp peer holds the identical [N,H]; take a distinct 1/tp_size slice so the
        # MoE runs once, not tp_size-redundantly. N need not divide tp_size at inference
        # (ragged prefill, small decode batch), so pad with zero rows and drop them after
        # the gather. Padding rows route+compute into their own (padding) token slots only,
        # never contaminating real tokens; Phase 11 slices them off.
        pad = (-N) % tp_size
        x_flat = hidden_states if pad == 0 else torch.cat(
            [hidden_states, hidden_states.new_zeros(pad, H)], dim=0)
        N_pad = N + pad
        local_N = N_pad // tp_size
        local_x = x_flat[tp_rank * local_N:(tp_rank + 1) * local_N]                  # [local_N, H]

        # ---- Phase 1: local router on the stripe (nano/HF routing, same math as C6) ----
        router_logits = self.gate(local_x)                                          # [local_N, E]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # ---- Phase 2: flatten to (token, expert, weight) records, sort by GLOBAL expert id ----
        token_ids = torch.arange(local_N, device=dev).repeat_interleave(self.top_k)  # [local_N*k]
        expert_ids = selected_experts.reshape(-1)                                   # GLOBAL ids
        weights = routing_weights.reshape(-1)
        sorted_eids, perm = torch.sort(expert_ids, stable=True)
        sorted_token_ids = token_ids[perm]
        sorted_weights = weights[perm]
        sorted_x = local_x[sorted_token_ids]                                        # [local_N*k, H], contiguous

        # ---- Phase 3-4: per-destination-rank send counts, negotiate recv counts ----
        # Each record goes to the ep rank owning its expert (global e // experts_per_rank).
        dest_ranks = sorted_eids // self.experts_per_rank
        send_counts = torch.bincount(dest_ranks, minlength=ep_size)                 # [ep_size], long
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=ep_group)            # negotiate splits
        send_list, recv_list = send_counts.tolist(), recv_counts.tolist()

        # ---- Phase 5: DISPATCH tokens + their global expert ids to the owning ranks ----
        recv_x = self._all_to_all_v(sorted_x, send_list, recv_list, ep_group)       # [M, H]
        recv_eids = self._all_to_all_v(sorted_eids, send_list, recv_list, ep_group)  # [M]

        # ---- Phase 6: re-sort received tokens by LOCAL expert id, build offsets ----
        # [contract RT1/RT2/RT5] over the received partition (o[0]=0, o[Eloc]=M, monotone).
        local_eids = recv_eids - self.expert_start                                  # -> [0, Eloc)
        local_sorted_eids, local_perm = torch.sort(local_eids, stable=True)
        local_sorted_x = recv_x[local_perm]
        counts = torch.bincount(local_sorted_eids, minlength=self.experts_per_rank)
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=dev),
            counts.cumsum(0),
        ])                                                                          # [Eloc + 1]

        # ---- Phase 7-8: local expert compute (shared box), then reverse the local sort ----
        # Guard ONLY the local GEMM (no collective inside); every collective below/above
        # stays unconditional so no rank skips one (NCCL deadlock).
        recv_out = torch.zeros_like(recv_x)
        if local_sorted_x.size(0) > 0:
            recv_out[local_perm] = self._expert_compute(local_sorted_x, offsets)

        # ---- Phase 9: COMBINE -- reverse all-to-all (swap the split lists) ----
        combined = self._all_to_all_v(recv_out, recv_list, send_list, ep_group)     # [local_N*k, H]

        # ---- Phase 10: weight-multiply + scatter back into this stripe's local output ----
        combined = combined * sorted_weights.unsqueeze(-1)
        local_y = torch.zeros_like(local_x)                                         # [local_N, H]
        local_y.index_add_(0, sorted_token_ids, combined)

        # ---- Phase 11: all_gather within tp_group -> replicated [N_pad, H]; drop padding ----
        y_flat = local_y.new_empty((N_pad, H))
        dist.all_gather_into_tensor(y_flat, local_y, group=tp_group)
        return y_flat[:N]


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
