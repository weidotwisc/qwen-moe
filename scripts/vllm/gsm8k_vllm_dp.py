"""GSM8K via *production vLLM*, matching nanovllm-weiz/gsm8k_moe.py's protocol so the
numbers are directly comparable to the nano-vLLM runs (only the engine differs).

Same 5-shot completion prompts, near-greedy (temperature 0.1, as nano forbids greedy),
max_tokens 256, and the identical strict/flexible answer extraction. Times the
generation (llm.generate) the same way nano's tqdm does. Config via env:

  DP  data_parallel_size      (default 8)
  TP  tensor_parallel_size    (default 1)
  EP  enable_expert_parallel  (default 1 -> ep_size = DP*TP)
  LIMIT / MAX_TOKENS          (default full 1319 / 256)

  CUDA_VISIBLE_DEVICES=0..7 DP=8 TP=1 EP=1 \
    scripts/vllm/.venv-vllm/bin/python scripts/vllm/gsm8k_vllm_dp.py
"""
import os
import re
import time
from glob import glob

from datasets import load_dataset
from vllm import LLM, SamplingParams

ANS_RE = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
NUM_RE = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")
STOP = ("\nQuestion:", "\n\nQuestion", "\n\n\n")


def norm(s: str) -> str:
    s = s.replace(",", "").strip()
    return s[:-2] if s.endswith(".0") else s


def gold(answer: str) -> str:
    m = ANS_RE.search(answer)
    return norm(m.group(1) if m else answer.split()[-1])


def extract(text: str):
    cut = len(text)
    for s in STOP:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    t = text[:cut]
    m = ANS_RE.search(t)
    strict = norm(m.group(1)) if m else None
    nums = NUM_RE.findall(t)
    flexible = norm(nums[-1]) if nums else None
    return strict, flexible


def resolve_model() -> str:
    snaps = glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*"))
    snaps = [s for s in snaps if os.path.exists(os.path.join(s, "config.json"))]
    assert snaps, "Qwen3-30B-A3B not in HF cache"
    return snaps[0]


def main():
    dp = int(os.environ.get("DP", 8))
    tp = int(os.environ.get("TP", 1))
    ep = os.environ.get("EP", "1") == "1"
    max_tokens = int(os.environ.get("MAX_TOKENS", 256))
    limit = os.environ.get("LIMIT", "")
    n_shot = 5

    ds = load_dataset("openai/gsm8k", "main")
    tr = ds["train"]
    shots = "".join(
        f"Question: {q}\nAnswer: {a}\n\n"
        for q, a in zip(tr["question"][:n_shot], tr["answer"][:n_shot])
    )
    te = ds["test"]
    if limit:
        te = te.select(range(int(limit)))
    prompts = [shots + f"Question: {q}\nAnswer:" for q in te["question"]]
    golds = [gold(a) for a in te["answer"]]

    print(f"[vllm-gsm8k] DP={dp} TP={tp} EP={ep} (ep_size={dp*tp if ep else 1})  "
          f"n={len(prompts)}  max_tokens={max_tokens}", flush=True)
    llm = LLM(resolve_model(), tensor_parallel_size=tp, data_parallel_size=dp,
              enable_expert_parallel=ep, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, enforce_eager=True)
    sp = SamplingParams(temperature=0.1, max_tokens=max_tokens)   # matches gsm8k_moe.py

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    gen_s = time.perf_counter() - t0

    strict_hit = flex_hit = 0
    for o, g in zip(outs, golds):
        s, fx = extract(o.outputs[0].text)
        strict_hit += (s == g)
        flex_hit += (fx == g)
    n = len(prompts)
    print(f"\n[vllm-gsm8k] DP={dp} TP={tp} EP={ep}  n={n}  "
          f"strict={strict_hit / n:.4f}  flexible={flex_hit / n:.4f}  "
          f"gen_time={gen_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
