import os
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.parallel import (
    init_parallel, get_tp_rank, get_tp_world_size, get_ep_world_size,
    get_replica_index, get_dp_leader_group, is_tp_leader,
)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # world = TP * DP == #GPUs == ep_size; tp_size = the TP subgroup degree.
        self.world_size = config.tensor_parallel_size * config.data_parallel_size
        self.tp_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        # [DP step 3] DP>1 requires the hybrid MoE schedule: the allreduce mode all-reduces
        # a [T,H] tensor over the world, and distinct per-replica batch sizes T -> NCCL shape
        # mismatch. Hybrid's ep collectives are count-negotiated all_to_alls, so replicas may
        # differ in phase/size. (gsm8k_moe already sets MOE_EP_MODE=hybrid for DP runs.)
        if config.data_parallel_size > 1:
            assert os.environ.get("MOE_EP_MODE") == "hybrid", (
                "DP>1 requires MOE_EP_MODE=hybrid (allreduce mode all-reduces [T,H] over the "
                "world -> shape mismatch across replicas with distinct batch sizes)")

        # Rendezvous port is env-overridable so multiple single-GPU instances can
        # run on one node (upstream hardcodes 2333; two instances collide -> EADDRINUSE).
        # One LLM instance's TP workers inherit this env, so they share a port; distinct
        # instances set distinct ports. Default 2333 keeps existing single-run behavior.
        port = int(os.environ.get("NANOVLLM_DIST_PORT", "2333"))
        dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        # [C7/C8 step 2] Build the TP/EP mesh BEFORE constructing the model (layers read
        # the parallel-state accessors at __init__). ep_group = the whole world (default,
        # None); tp_group = this rank's block of tp_size contiguous ranks. DP==1 keeps
        # tp_group=None (the world) -> byte-identical to the old flat path.
        if config.data_parallel_size == 1:
            my_ranks, my_tp_group = list(range(self.world_size)), None
            dp_leader_group, replica_index = None, 0
        else:
            tp_groups = []
            for j in range(config.data_parallel_size):
                rs = list(range(j * self.tp_size, (j + 1) * self.tp_size))
                tp_groups.append((rs, dist.new_group(rs)))   # ALL ranks call new_group for EVERY group, same order
            my_ranks, my_tp_group = tp_groups[rank // self.tp_size]
            # [DP step 3] DP-leader group = tp_rank-0 of every replica; used to gather each
            # replica's sampled tokens back to rank 0. All ranks call new_group, AFTER the
            # tp_groups loop, so the collective ordering matches on every rank.
            dp_leader_group = dist.new_group(list(range(0, self.world_size, self.tp_size)))
            replica_index = rank // self.tp_size
        init_parallel(tp_group=my_tp_group, ep_group=None, tp_ranks=my_ranks,
                      dp_leader_group=dp_leader_group, replica_index=replica_index)
        print(f"[mesh] rank={rank} world={self.world_size} tp_group={my_ranks} "
              f"tp_rank={get_tp_rank()} tp_size={get_tp_world_size()} ep_size={get_ep_world_size()}", flush=True)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        arch = hf_config.architectures[0]
        model_cls = Qwen3MoeForCausalLM if arch == "Qwen3MoeForCausalLM" else Qwen3ForCausalLM
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # [DP step 3] the run_dp payload carries DP sub-batches, so scale the buffer.
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20 * max(1, config.data_parallel_size))
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= len(self.shm.buf), f"shm payload {n+4}B exceeds buffer {len(self.shm.buf)}B ({method_name})"
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # [C4] KV replication: tp>num_kv_heads -> 1 (replicated) head per rank, so the
        # cache is sized for the per-rank head count (matches Qwen3Attention.num_kv_heads).
        # [C7/C8 step 2] KV heads shard within the TP group, not the whole world.
        num_kv_heads = max(1, hf_config.num_key_value_heads // self.tp_size)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        local_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert local_blocks > 0, f"rank {self.rank}: no room for KV cache"
        # [C7/C8 mesh fix] num_kvcache_blocks is computed per-rank from LOCAL free memory,
        # but the rank-0 scheduler hands out block-ids from ITS count to ALL ranks. If any
        # rank sized a smaller cache, a scheduled slot overflows it -> store_kvcache writes
        # out of bounds -> CUDA illegal memory access on that rank. Reconcile to the GLOBAL
        # MIN so every rank's cache holds any block the scheduler can assign. (Invisible
        # under sharded TP where all ranks match; exposed at tp=1, where warmup MoE-dispatch
        # imbalance diverges peak memory -> divergent block counts across ranks.)
        if self.world_size > 1:
            t = torch.tensor([local_blocks], dtype=torch.int64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            config.num_kvcache_blocks = int(t.item())
        else:
            config.num_kvcache_blocks = local_blocks
        print(f"[kvblocks rank={self.rank}] local={local_blocks} reconciled(min)={config.num_kvcache_blocks}", flush=True)
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # [DP step 3] each replica's LEADER samples its own sub-batch (was rank-0-only). At
        # DP=1 tp_group spans the world so is_tp_leader() == (global rank 0) -> unchanged.
        leader = is_tp_leader()
        temperatures = self.prepare_sample(seqs) if leader else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if leader else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_dummy(self):
        # [DP step 3] Lockstep filler for a drained/idle replica: run a full model forward on
        # one synthetic token so this replica's ranks still post every ep_group MoE collective
        # (48 layers) in step with the working replicas. Touches no KV (slot=-1 -> store_kvcache
        # no-ops) and does no sampling. tp_group collectives (embed/o_proj/all_gather) are
        # per-replica, so skipping the lm_head gather here is fine (it's not run).
        input_ids = torch.tensor([0], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([0], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu = torch.tensor([0, 1], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor([-1], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu, cu, 1, 1, slot_mapping, None, None)
        self.model(input_ids, positions)   # embed + all MoE layers -> every ep/tp collective
        reset_context()

    def run_dp(self, payload: list):
        # [DP step 3] One synchronized global step across DP replicas. `payload[j]` is this
        # step's (sub_seqs, is_prefill) for replica j; an empty sub_seqs means replica j is
        # drained -> it runs a dummy forward to stay in ep-collective lockstep. Each replica's
        # LEADER samples its shard; leaders gather their token lists to rank 0.
        j = get_replica_index()
        sub_seqs, is_prefill = payload[j]
        if sub_seqs:
            token_ids = self.run(sub_seqs, is_prefill)   # leader: list; non-leaders: None
        else:
            self.run_dummy()
            token_ids = []
        if is_tp_leader():
            dp = self.config.data_parallel_size
            gathered = [None] * dp if self.rank == 0 else None
            # dst=0 is both global rank 0 and dp-leader-group rank 0 (leader_ranks[0]==0),
            # so it is unambiguous regardless of gather_object's dst convention.
            dist.gather_object(token_ids if token_ids is not None else [], gathered, dst=0, group=get_dp_leader_group())
            if self.rank == 0:
                return gathered      # gathered[j] == replica j's tokens (group order == replica order)
        return None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
