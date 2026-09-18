"""GSM8K accuracy for the Qwen3-MoE integration (loop vs fused), single GPU.

Loads GSM8K directly (cached via `datasets`), builds a standard 5-shot prompt,
runs every question through nano-vLLM's generate (near-greedy, progress bar over
all rows), extracts the answer, and reports strict + flexible exact-match. Pick
the kernel with MOE_KERNEL=loop|fused.

    cd nanovllm-weiz
    MOE_KERNEL=loop  python gsm8k_moe.py
    MOE_KERNEL=fused python gsm8k_moe.py

Env: LIMIT (#questions, blank=all 1319), MAX_TOKENS (256), N_SHOT (5).

Self-contained harness (standard 5-shot, near-greedy -- nano-vLLM forbids
temperature=0). Approximately comparable to the lm-eval/vLLM oracle
(~0.8923 strict), not bit-identical; the integration signal is loop == fused,
both in the ~0.89 ballpark.
"""
import os
import re
from glob import glob

from datasets import load_dataset
from nanovllm import LLM, SamplingParams

ANS_RE = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
NUM_RE = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")
STOP = ("\nQuestion:", "\n\nQuestion", "\n\n\n")


def norm(s: str) -> str:
    s = s.replace(",", "").strip()
    return s[:-2] if s.endswith(".0") else s


def gold(answer: str) -> str:
    m = ANS_RE.search(answer)
    return norm(m.group(1) if m else answer.split()[-1])


def resolve_model() -> str:
    snaps = glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*"))
    snaps = [s for s in snaps if os.path.exists(os.path.join(s, "config.json"))]
    assert snaps, "Qwen3-30B-A3B not in HF cache"
    return snaps[0]


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


def main():
    kernel = os.environ.get("MOE_KERNEL", "loop")
    n_shot = int(os.environ.get("N_SHOT", 5))
    max_tokens = int(os.environ.get("MAX_TOKENS", 256))
    limit = os.environ.get("LIMIT", "")
    tp = int(os.environ.get("TP", 1))   # tensor_parallel_size; == ep_size for C6 (DP=1)

    ds = load_dataset("openai/gsm8k", "main")
    tr = ds["train"]                                   # column access -> list[str]
    shots = "".join(
        f"Question: {q}\nAnswer: {a}\n\n"
        for q, a in zip(tr["question"][:n_shot], tr["answer"][:n_shot])
    )
    te = ds["test"]
    if limit:
        te = te.select(range(int(limit)))
    prompts = [shots + f"Question: {q}\nAnswer:" for q in te["question"]]
    golds = [gold(a) for a in te["answer"]]

    print(f"[gsm8k] MOE_KERNEL={kernel}  TP(=ep)={tp}  n={len(prompts)}  n_shot={n_shot}  max_tokens={max_tokens}", flush=True)
    llm = LLM(resolve_model(), enforce_eager=True, tensor_parallel_size=tp, max_model_len=4096)
    sp = SamplingParams(temperature=0.1, max_tokens=max_tokens)   # near-greedy (nano-vLLM forbids temp=0)
    outs = llm.generate(prompts, sp)   # progress bar over len(prompts)

    strict_hit = flex_hit = 0
    for o, g in zip(outs, golds):
        s, fx = extract(o["text"])
        strict_hit += (s == g)
        flex_hit += (fx == g)
    n = len(prompts)
    print(f"\n[gsm8k] kernel={kernel}  n={n}  "
          f"strict={strict_hit / n:.4f}  flexible={flex_hit / n:.4f}", flush=True)


if __name__ == "__main__":
    main()
