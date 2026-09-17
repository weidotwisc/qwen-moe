"""Single-GPU smoke test for the Qwen3-MoE adaptation in nanovllm-weiz.

Loads Qwen3-30B-A3B on ONE GPU (tp=1, eager) and prints completions for a few
chat prompts. Purpose: verify the model LOADS and the MoE forward produces
COHERENT text (not garbage) before wiring up the GSM8K harness. Speed is NOT
the point (the correctness-first per-expert loop is slow).

Run with the repo venv (torch 2.11), from this directory, pinned to one GPU:

    cd nanovllm-weiz
    CUDA_VISIBLE_DEVICES=0 \
      /gpfs/users/weiz/workspace/personal/qwen-moe/.venv/bin/python smoke_moe.py

Optionally pass a model dir as argv[1]; default resolves the cached snapshot.
"""
import os
import sys
from glob import glob

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def resolve_model(argv) -> str:
    if len(argv) > 1:
        return argv[1]
    # Default: the cached Qwen3-30B-A3B snapshot (model files resolved straight
    # from the HF hub cache -- no dependency on any sibling checkout).
    snaps = glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*"))
    snaps = [s for s in snaps if os.path.exists(os.path.join(s, "config.json"))]
    assert snaps, "Qwen3-30B-A3B not found in HF cache; pass a model dir as argv[1]"
    return snaps[0]


def main():
    path = resolve_model(sys.argv)
    print(f"[smoke] model = {path}", flush=True)
    print(f"[smoke] MOE_KERNEL = {os.environ.get('MOE_KERNEL', 'loop')}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(path)
    # enforce_eager=True is REQUIRED here: the MoE forward has host syncs and
    # data-dependent block shapes that CUDA-graph capture cannot handle.
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # temperature must be > 0 (nano-vLLM forbids greedy); keep it low so the
    # coherence check is near-deterministic.
    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)

    raw_prompts = [
        "Introduce yourself in one sentence.",
        "What is 12 * 8? Answer with just the number.",
        "List the first five prime numbers.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]

    outputs = llm.generate(prompts, sampling_params)
    for p, o in zip(raw_prompts, outputs):
        print("\n" + "=" * 60)
        print(f"Prompt:     {p}")
        print(f"Completion: {o['text']!r}")


if __name__ == "__main__":
    main()
