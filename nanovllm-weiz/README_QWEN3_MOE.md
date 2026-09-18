# Qwen3-30B-A3B (Qwen3-MoE) support for nano-vLLM

This fork adds **Qwen3 Mixture-of-Experts** inference (e.g. Qwen3-30B-A3B) to
nano-vLLM — single-GPU **and** expert-parallel (multi-GPU, DP=1).

**Base:** clean checkout of upstream nano-vLLM
(`github.com/GeeeekExplorer/nano-vllm`) at commit
**`bb823b3e06983d71485a8e1f23715ebd87d98ef8`**
("Merge pull request #218 from GeeeekExplorer/chunked-prefill-refactor",
2026-04-26). That pristine base is committed on its own (qwen-moe commit
`aba5ea76`) so this feature is a clean diff on top of it —
`git diff aba5ea76 HEAD` is the exact delta.

Stock nano-vLLM supports **dense** Qwen3 only. Two deltas make Qwen3-MoE run at
scale: (1) the FFN becomes a **sparse MoE block**, and (2) GQA attention must
handle `tp_size > num_kv_heads` (Qwen3-30B-A3B has 4 KV heads, so tp=8 needs
**KV-head replication**). Everything else — QK-norm, RoPE, RMSNorm, embeddings —
is reused from `models/qwen3.py`.

## What changed

| File | Change |
|---|---|
| `nanovllm/models/qwen3_moe.py` | **new** — `Qwen3MoeForCausalLM` + `Qwen3MoeSparseMoeBlock`: router + stacked experts, two swappable kernels, and **[C6] expert parallelism** (filter-to-local experts + `all_reduce` combine, per-rank expert shard). |
| `nanovllm/layers/fused_moe.py` | **new** — fused MoE Triton grouped-GEMM (vendored from `bootcamp/ex09_fused_moe`). |
| `nanovllm/utils/loader.py` | expert-weight routing: parses the global expert id and dispatches to the block's stacked param loader (the per-rank EP shard-map/skip lives in `qwen3_moe.py`). |
| `nanovllm/layers/linear.py` | **[C4]** `QKVParallelLinear` — GQA + **KV-head replication** when `tp_size > num_kv_heads` (sizing + replication-aware weight loading). |
| `nanovllm/models/qwen3.py` | **[C4]** `Qwen3Attention` allows `tp_size > num_kv_heads` (`num_kv_heads = max(1, …)`). |
| `nanovllm/engine/model_runner.py` | dispatch model class on `architectures[0]`; **[C4]** KV-cache sized for the per-rank (replicated) head count; env-overridable NCCL rendezvous port (`NANOVLLM_DIST_PORT`) for same-node multi-instance. |
| `smoke_moe.py` | **new** — single-GPU coherence smoke test. |
| `gsm8k_moe.py` | **new** — GSM8K accuracy harness (loop/fused × any `TP`), the quantitative integration test. |

## MoE block: two expert kernels on one representation

Experts are stored **stacked** (`w_gate/w_up/w_down`, shapes
`[E,I,H] / [E,I,H] / [E,H,I]`, `E` = experts held by this rank) — the single
layout shared by both compute paths, selected by the `MOE_KERNEL` env var:

- `MOE_KERNEL=loop` (default) — per-expert grouped compute (reference path).
- `MOE_KERNEL=fused` — fused Triton grouped-GEMM (~5× faster on the batched GSM8K run).

Routing is identical either way: router → top-k → renormalize → *(filter to local
experts under EP)* → sort by expert → offset array → grouped compute → weight +
unpermute + scatter-combine → *(one `all_reduce` under EP)*.

## Expert parallelism (C6, DP=1) + GQA KV replication (C4)

**[C6] — `bootcamp/ex06_ep/solution_lean.py`.** Launch with
`tensor_parallel_size = N` and the MoE runs expert-parallel with
**`ep_size = N` (== `tp_size`, one flat group; "DP=1")**. nano-vLLM's TP already
replicates the batch across ranks — exactly the replicated-input precondition the
lean schedule needs. Each rank holds `num_experts / ep_size` stacked experts, the
router runs globally on every rank, records are **filtered to this rank's local
experts**, computed (loop or fused), and a single **`all_reduce`** sums the
per-rank partials into the full output. `ep_size == 1` is the original single-GPU
path, byte-identical.

**[C4] — `bootcamp/ex04_gqa_tp/solution.py`.** Qwen3-30B-A3B has 4 KV heads, so
`tp_size > 4` cannot give each rank its own KV head. C4 **replicates** each KV
head across `num_kv_replicas = max(1, tp_size // num_kv_heads)` ranks (rank `r`
holds KV head `r // num_kv_replicas`); Q shards normally. It touches
`QKVParallelLinear` (output sizing + replication-aware K/V weight loading),
`Qwen3Attention` (relaxed divisibility assert), and the KV-cache sizing in
`model_runner`. Without it, tp=8 dies at `divide(4, 8)`. flash-attn handles the
resulting GQA (`num_heads > num_kv_heads`) natively — no `repeat_interleave`
needed.

## Run

Use the repo's uv `.venv` (torch 2.11 + triton 3.6 + transformers + datasets).
From this directory:

```sh
# single GPU (ep=1)
CUDA_VISIBLE_DEVICES=0 python smoke_moe.py                       # loop (coherence)
CUDA_VISIBLE_DEVICES=0 MOE_KERNEL=fused python smoke_moe.py      # fused

# GSM8K accuracy (loop/fused, any TP == ep); ep ∈ {1,2,4,8} valid (128 % ep == 0)
CUDA_VISIBLE_DEVICES=0               MOE_KERNEL=fused python gsm8k_moe.py           # ep=1
CUDA_VISIBLE_DEVICES=0,1,2,3    TP=4 MOE_KERNEL=fused python gsm8k_moe.py           # ep=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=8 MOE_KERNEL=fused python gsm8k_moe.py      # ep=8 (needs C4)
```

`enforce_eager=True` is required (the MoE forward has host syncs and
data-dependent block shapes CUDA-graph capture can't handle); the scripts set it.
Run several instances on one node with distinct `NANOVLLM_DIST_PORT`.

## Traceability

Greppable tags map each part back to the verified bootcamp components (the
"generated code → contracts" thread):

- `[C4]`   → `bootcamp/ex04_gqa_tp/solution.py` — GQA + KV-head replication under TP.
- `[C5-b]` → `bootcamp/ex05_moe_baseline/reference_b.py` — sorted-offset routing + loop expert path.
- `[C6]`   → `bootcamp/ex06_ep/solution_lean.py` — lean expert parallelism (DP=1): filter-to-local + one `all_reduce`.
- `[C9]`   → `bootcamp/ex09_fused_moe/solution.py` — fused grouped-GEMM kernel.
- `[contract RT1/RT2/RT5]` → ex05 routing-partition contracts on `offsets` (now over the **local** partition under EP).

```sh
grep -rn "\[C4\]\|\[C5-b\]\|\[C6\]\|\[C9\]\|\[contract" nanovllm/
```

## Scope / status

- **Single-GPU and expert-parallel (C6, DP=1 / `all_reduce`)** across
  `ep ∈ {1,2,4,8}` (`tp == ep == world_size`). C7 (TP=1 / dispatch) and C8
  (hybrid) are not yet wired.
- **Validated quantitatively** against the production-vLLM GSM8K oracle
  (strict-match **0.8923**): nano-vLLM strict lands **0.8886–0.8969 across
  ep=1–8**, loop and fused, all within the oracle's ±0.85% stderr. `loop == fused`
  and `ep=1 ≡ ep=4 ≡ ep=8` → the MoE integration, KV replication, and `all_reduce`
  combine are correct.
- **Speed:** fused ≫ loop; EP throughput peaks around **ep=4** for this
  model+batch (ep=8 is past the knee — the `all_reduce` + expert-load-imbalance
  straggler grow faster than per-rank compute shrinks). vLLM stays ~2–3× faster on
  generation (mature engine, tuned/graph-able MoE kernel); closing that is future
  work, and the kernel is a swappable verified component.
```
