import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # world = TP * DP ranks (== #GPUs); DP==1 -> tensor_parallel_size, unchanged.
        world_size = config.tensor_parallel_size * config.data_parallel_size
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # [DP step 3] one Scheduler per DP replica (each is a self-contained per-replica unit:
        # its own waiting/running/block_manager). DP==1 keeps the single scheduler untouched.
        self.dp = config.data_parallel_size
        if self.dp == 1:
            self.scheduler = Scheduler(config)
        else:
            self.schedulers = [Scheduler(config) for _ in range(self.dp)]
            self._rr = 0
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if self.dp == 1:
            self.scheduler.add(seq)
        else:
            # [DP step 3] static round-robin: a request lives on one replica for its lifetime
            # (KV never migrates), like a DistributedSampler shard fixed at admission.
            self.schedulers[self._rr % self.dp].add(seq)
            self._rr += 1

    def step(self):
        if self.dp == 1:
            seqs, is_prefill = self.scheduler.schedule()
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            return outputs, num_tokens
        # [DP step 3] synchronized global step: schedule every replica (a drained replica gets
        # an empty sub-batch -> a dummy forward keeps it in ep-collective lockstep), run all in
        # one lockstep forward, gather per-replica tokens on rank 0, then postprocess each.
        payload = [([], False) if s.is_finished() else s.schedule() for s in self.schedulers]
        token_lists = self.model_runner.call("run_dp", payload)   # rank 0 -> list indexed by replica
        outputs = []
        prefill_toks = decode_seqs = 0
        for j, s in enumerate(self.schedulers):
            sub_seqs, is_prefill = payload[j]
            if not sub_seqs:                    # dummy replica this step
                continue
            s.postprocess(sub_seqs, token_lists[j], is_prefill)
            outputs += [(seq.seq_id, seq.completion_token_ids) for seq in sub_seqs if seq.is_finished]
            if is_prefill:
                prefill_toks += sum(seq.num_scheduled_tokens for seq in sub_seqs)
            else:
                decode_seqs += len(sub_seqs)
        num_tokens = prefill_toks if prefill_toks else -decode_seqs   # tqdm throughput display only
        return outputs, num_tokens

    def is_finished(self):
        if self.dp == 1:
            return self.scheduler.is_finished()
        return all(s.is_finished() for s in self.schedulers)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
