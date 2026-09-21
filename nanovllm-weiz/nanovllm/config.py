import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1        # TP degree = size of each tp_group
    data_parallel_size: int = 1          # [C7/C8 step 2] # of tp_groups (replicas); world = TP*DP
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        # world = TP * DP = #ranks = #GPUs you launch on (== ep_size). No fixed ceiling:
        # the real bounds live where they belong -- world is capped by the GPUs you make
        # visible (single-node localhost rendezvous), num_experts % world by the MoE
        # block, and KV/head divisibility by the attention layer. DP == 1 is the flat default.
        assert 1 <= self.tensor_parallel_size and 1 <= self.data_parallel_size
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
