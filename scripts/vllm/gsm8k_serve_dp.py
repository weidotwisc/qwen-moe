"""GSM8K against a `vllm serve` DP+EP server (native DP=DP, TP=TP, EP=DP*TP).

vLLM blocks single-process offline `LLM(data_parallel_size>1)`; the supported native
DP+EP path is the online server (`vllm serve ... -dp N --enable-expert-parallel`), which
launches one engine per DP rank behind a coordinator. This driver: launches that server,
waits for /health, fires the SAME 1319 GSM8K 5-shot prompts as nanovllm-weiz/gsm8k_moe.py
(temp 0.1, max_tokens 256, same strict/flexible extraction) concurrently at
/v1/completions, times the request phase (server startup excluded), then tears the
server down. Comparable to the nano runs modulo the HTTP client + server continuous
batching.

  CUDA_VISIBLE_DEVICES=0..7 DP=8 TP=1 \
    scripts/vllm/.venv-vllm/bin/python scripts/vllm/gsm8k_serve_dp.py
Env: DP(8) TP(1) LIMIT MAX_TOKENS(256) PORT(8007) CONC(256).
"""
import os
import re
import time
import signal
import asyncio
import subprocess
import urllib.request
from glob import glob

import aiohttp
from datasets import load_dataset

ANS_RE = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
NUM_RE = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")
STOP = ("\nQuestion:", "\n\nQuestion", "\n\n\n")


def norm(s: str) -> str:
    s = s.replace(",", "").strip()
    return s[:-2] if s.endswith(".0") else s


def gold(a: str) -> str:
    m = ANS_RE.search(a)
    return norm(m.group(1) if m else a.split()[-1])


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


def wait_ready(port: int, proc: subprocess.Popen, timeout: int = 900):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode}); see /tmp/vllm_server.log")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready within timeout")


async def _fire(session, url, name, prompt, max_tokens, sem, results, i, prog):
    body = {"model": name, "prompt": prompt, "temperature": 0.1, "max_tokens": max_tokens}
    async with sem:
        async with session.post(url, json=body) as r:
            j = await r.json()
    results[i] = j["choices"][0]["text"]
    prog[0] += 1
    if prog[0] % 200 == 0:
        print(f"[client] {prog[0]}/{len(results)}", flush=True)


async def run_all(port: int, name: str, prompts, max_tokens: int, conc: int):
    """Fire all prompts with at most `conc` in flight, on a single event loop (no thread
    per request). Returns (results, gen_s) timed around the request phase only."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    results = [None] * len(prompts)
    prog = [0]
    sem = asyncio.Semaphore(conc)
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=conc + 32, limit_per_host=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(
            _fire(session, url, name, p, max_tokens, sem, results, i, prog))
            for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)
        gen_s = time.perf_counter() - t0
    return results, gen_s


def scrape(port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            return r.read().decode()
    except Exception:
        return ""


# Prometheus line matchers: histogram `_sum`/`_count`, and `_total` counters. Each vLLM
# DP engine exports its own label set (engine="k"), so we take the max value seen per
# (metric,label) across a burst of scrapes (cumulative/monotone) then SUM across labels.
_CNT = re.compile(r'^(vllm:[a-zA-Z_]+)_count(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$', re.M)
_SUM = re.compile(r'^(vllm:[a-zA-Z_]+)_sum(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$', re.M)
_TOT = re.compile(r'^(vllm:[a-zA-Z_]+_total)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$', re.M)


def _latest(snaps, pat):
    best = {}
    for txt in snaps:
        for m in pat.finditer(txt):
            k = (m.group(1), m.group(2) or "")
            try:
                v = float(m.group(3))
            except ValueError:
                continue
            if k not in best or v > best[k]:
                best[k] = v
    return best


def parse_metrics(snaps):
    counts, sums, totals = (_latest(snaps, p) for p in (_CNT, _SUM, _TOT))
    hist = {}   # base -> (mean, n, sum) aggregated over DP engines
    for b in {b for (b, _) in counts}:
        c = sum(v for (bb, _), v in counts.items() if bb == b)
        s = sum(v for (bb, _), v in sums.items() if bb == b)
        if c > 0 and any(bb == b for (bb, _) in sums):
            hist[b] = (s / c, int(c), s)
    ctr = {n: sum(v for (nn, _), v in totals.items() if nn == n)
           for n in {n for (n, _) in totals}}
    n_labels = len({lbl for (_, lbl) in counts})   # DP-shape diagnostic
    return hist, ctr, n_labels


def report_metrics(snaps, gen_s):
    hist, ctr, n_labels = parse_metrics(snaps)
    gen_tok = ctr.get("vllm:generation_tokens_total", 0.0)
    prm_tok = ctr.get("vllm:prompt_tokens_total", 0.0)
    print("\n[metrics] server-side (HTTP + client tail excluded), aggregated over DP engines")
    print(f"  distinct label-sets={n_labels}  prompt_tokens={int(prm_tok)}  "
          f"generation_tokens={int(gen_tok)}")

    def show(tag, base, in_ms=False):
        if base in hist:
            mean, n, _ = hist[base]
            val = f"{mean * 1000:.1f}ms" if in_ms else f"{mean:.3f}s"
            print(f"    {tag:<36} mean={val}  (n={n})")

    show("per-req INFERENCE (prefill+decode)", "vllm:request_inference_time_seconds")
    show("  prefill", "vllm:request_prefill_time_seconds")
    show("  decode", "vllm:request_decode_time_seconds")
    show("per-req QUEUE (wait, excl. compute)", "vllm:request_queue_time_seconds")
    show("per-req E2E (queue+inference)", "vllm:e2e_request_latency_seconds")
    show("TTFT (time to first token)", "vllm:time_to_first_token_seconds", in_ms=True)
    show("TPOT (per output token)", "vllm:time_per_output_token_seconds", in_ms=True)
    if gen_s > 0:
        print(f"    aggregate generation throughput   ~{gen_tok / gen_s:.0f} tok/s  "
              f"(8 replicas; floor — client window incl. ramp+tail)")


def main():
    dp = int(os.environ.get("DP", 8))
    tp = int(os.environ.get("TP", 1))
    max_tokens = int(os.environ.get("MAX_TOKENS", 256))
    limit = os.environ.get("LIMIT", "")
    port = int(os.environ.get("PORT", "8007"))
    conc = int(os.environ.get("CONC", "256"))
    name = "qwen3"
    model = resolve_model()

    ds = load_dataset("openai/gsm8k", "main")
    tr = ds["train"]
    shots = "".join(f"Question: {q}\nAnswer: {a}\n\n"
                    for q, a in zip(tr["question"][:5], tr["answer"][:5]))
    te = ds["test"]
    if limit:
        te = te.select(range(int(limit)))
    prompts = [shots + f"Question: {q}\nAnswer:" for q in te["question"]]
    golds = [gold(a) for a in te["answer"]]

    cmd = [".venv-vllm/bin/vllm", "serve", model, "--served-model-name", name,
           "-tp", str(tp), "-dp", str(dp), "--enable-expert-parallel",
           "--dtype", "bfloat16", "--max-model-len", "4096", "--enforce-eager",
           "--gpu-memory-utilization", "0.9", "--port", str(port)]
    print("[serve] " + " ".join(cmd), flush=True)
    slog = open("/tmp/vllm_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT)
    try:
        wait_ready(port, proc)
        print(f"[serve] ready; sending {len(prompts)} reqs at conc={conc}", flush=True)
        results, gen_s = asyncio.run(run_all(port, name, prompts, max_tokens, conc))
        # Burst-scrape /metrics while the server is still up. Cumulative histograms hold all
        # 1319 requests; the burst ensures every DP engine's label set is captured even if
        # the shared port round-robins across the 8 API servers.
        snaps = [scrape(port) for _ in range(60)]
        strict = flex = 0
        for txt, g in zip(results, golds):
            s, fx = extract(txt or "")
            strict += (s == g)
            flex += (fx == g)
        n = len(prompts)
        print(f"\n[vllm-serve] DP={dp} TP={tp} EP=on  n={n}  "
              f"strict={strict / n:.4f}  flexible={flex / n:.4f}  gen_time={gen_s:.1f}s", flush=True)
        report_metrics(snaps, gen_s)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        slog.close()


if __name__ == "__main__":
    main()
