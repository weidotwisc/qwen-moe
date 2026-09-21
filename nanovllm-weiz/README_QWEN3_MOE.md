# Qwen3-30B-A3B (Qwen3-MoE) support for nano-vLLM

This fork adds **Qwen3 Mixture-of-Experts** inference (e.g. Qwen3-30B-A3B) to
nano-vLLM — single-GPU **and** expert-parallel (multi-GPU) over a **TP/EP device
mesh** (`world = TP × DP`), with two MoE expert-parallel schedules (`all_reduce`
and hybrid dispatch).

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
| `nanovllm/models/qwen3_moe.py` | **new** — `Qwen3MoeForCausalLM` + `Qwen3MoeSparseMoeBlock`: router + stacked experts, two swappable kernels (`MOE_KERNEL`), and two EP schedules (`MOE_EP_MODE`): **[C6]** filter-to-local + `all_reduce`, and **[C7/C8]** hybrid stripe → all-to-all dispatch → combine → all_gather (`_forward_hybrid`). |
| `nanovllm/layers/fused_moe.py` | **new** — fused MoE Triton grouped-GEMM (vendored **verbatim** from `bootcamp/ex09_fused_moe`). Has an **OPEN `TP=1` crash** + a latent int32 overflow (>~1.05M rows) — see below; **not patched here** (fix + re-verify in ex09, then re-sync). |
| `nanovllm/utils/loader.py` | expert-weight routing: parses the global expert id and dispatches to the block's stacked param loader (the per-rank EP shard-map/skip lives in `qwen3_moe.py`). |
| `nanovllm/layers/linear.py` | **[C4]** `QKVParallelLinear` — GQA + **KV-head replication** when `tp_size > num_kv_heads` (sizing + replication-aware weight loading). |
| `nanovllm/models/qwen3.py` | **[C4]** `Qwen3Attention` allows `tp_size > num_kv_heads` (`num_kv_heads = max(1, …)`). |
| `nanovllm/engine/model_runner.py` | dispatch model class on `architectures[0]`; **[C4]** KV-cache sized for the per-rank (replicated) head count; env-overridable NCCL rendezvous port (`NANOVLLM_DIST_PORT`) for same-node multi-instance. |
| `smoke_moe.py` | **new** — single-GPU coherence smoke test. |
| `gsm8k_moe.py` | **new** — GSM8K accuracy harness (loop/fused × any `TP`/`DP`), the quantitative integration test. |
| `nanovllm/utils/parallel.py` | **new [C7/C8 mesh]** — TP/EP device mesh + accessors (`get_tp_*`/`get_ep_*`); `world = TP × DP = #GPUs = ep_size`. |
| `nanovllm/config.py`, `engine/llm_engine.py`, `engine/model_runner.py` | **[C7/C8 mesh]** `data_parallel_size` knob; spawn `world = TP×DP` ranks; build `tp_group`/`ep_group`; KV sized by `tp_size`. |
| `nanovllm/layers/linear.py`, `layers/embed_head.py`, `models/qwen3.py` | **[C7/C8 mesh]** shard / all_reduce / LM-head gather over `tp_group` via the accessors (was the flat world). |
| `repro_fused_moe_bug.py` | **new** — single-GPU reproducer + fused-vs-loop equivalence check for the [C9] int32 overflow. |

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

## Hybrid dispatch + device mesh (C7/C8)

**[C7/C8] — `bootcamp/ex07_tp_ep_hybrid/solution.py`.** `MOE_EP_MODE=hybrid`
selects a second EP schedule. Instead of routing the whole batch on every rank and
summing with `all_reduce` (C6), the ranks **stripe** the batch within their
`tp_group` (`1/tp_size` each), **dispatch** every token to the rank owning its
expert across the `ep_group` (`all_to_all`), run the local experts, **combine**
back, and `all_gather` within the `tp_group` to rebuild the output
(`_forward_hybrid`). It reuses the same stacked-weight `_expert_compute` box as C6
and nano/HF routing (fp32 softmax → top-k → renorm). Collectives are unconditional
(no rank skips one → no NCCL deadlock); `N` is zero-padded to a multiple of
`tp_size` and the padding sliced off after the gather.

**Device mesh (`nanovllm/utils/parallel.py`).** The engine is no longer a flat
world: `world = tensor_parallel_size (TP) × data_parallel_size (DP) = #GPUs = ep_size`.
- `tp_group` — `TP` contiguous ranks; attention / dense linears / embedding / LM
  head, and the MoE stripe + all_gather.
- `ep_group` — the whole world; experts shard across ALL ranks, so only the MoE
  dispatch/combine cross `tp_group` boundaries.
Layers read `get_tp_*()` / `get_ep_*()` accessors (set once in `model_runner`)
instead of `dist.get_world_size()/get_rank()`. **`DP=1` keeps `tp_group=None` (the
world) → byte-identical to the old flat path.**

`tp_size < world` is now expressible: `TP=4 DP=2` is the **C8** shape (two TP groups
over one EP world), `TP=1 DP=8` is **C7** (pure EP). NOTE: there is **no DP engine
yet** (step 3) — the batch is still **replicated** across DP replicas, so `DP>1`
builds and validates the subgroup collectives but the replicas do redundant work
until step 3 partitions the batch. `MOE_EP_MODE=allreduce` (C6) stays the default;
`ep_size==1` is the single-GPU path.

### [C9] fused-kernel crash under `TP=1` — OPEN

`MOE_EP_MODE=hybrid MOE_KERNEL=fused` at **`TP=1 DP=8`** full-scale GSM8K crashes with
a **CUDA illegal memory access**; `MOE_KERNEL=loop` at the same config completes
correctly (strict **0.8908**). So the fault is isolated to the C9 fused Triton kernel,
and it is **data-dependent** (crashes in prefill or decode across runs, only at full
scale — never at `LIMIT=128`). **Root cause not yet found; the crash is OPEN.**

What we ruled out: a real **int32 overflow** exists in `x_ptr + e_start * stride_xm`
(with `e_start` int32, `stride_xm == H == 2048`) once `e_start` exceeds ~2³¹/2048 ≈
**1.05M rows** — `repro_fused_moe_bug.py` reproduces it deterministically at M≈1.57M,
and an **int64 base-pointer widening** removes it (verified in the repro). That fix is
**not applied here** — the kernel stays **verbatim to verified ex09**, so the fix +
re-verification belong in ex09, then re-sync. **And that latent overflow is NOT the
tp=1 crash:** at `TP=1` a rank receives at most the total dispatched (~8·16384·8 ≈
1.05M rows), where `e_start·2048` stays just *under* 2³¹ — and the uniform M=1.05M
repro does **not** crash. The real tp=1 fault is a **different, lower-M,
data-pattern-specific** bug still to be found (next: dump the exact `M`/offsets/
per-expert counts before the failing `fused_moe_forward`, build a faithful repro, run
`compute-sanitizer`). For a correctness-focused artifact this matters: an OOB that
crashes at tp=1 could silently corrupt at another shape, so it needs a real fix.

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

# hybrid dispatch schedule (MOE_EP_MODE=hybrid); world = TP*DP GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=8       MOE_KERNEL=fused MOE_EP_MODE=hybrid python gsm8k_moe.py  # DP=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=4 DP=2  MOE_KERNEL=fused MOE_EP_MODE=hybrid python gsm8k_moe.py  # C8 shape
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=1 DP=8  MOE_KERNEL=fused MOE_EP_MODE=hybrid python gsm8k_moe.py  # C7 shape
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
- `[C7/C8]` → `bootcamp/ex07_tp_ep_hybrid/solution.py` — hybrid stripe/gather (tp) + all-to-all dispatch/combine (ep); `_forward_hybrid`, step 1 at `tp==ep==world` (DP=1).
- `[C9]`   → `bootcamp/ex09_fused_moe/solution.py` — fused grouped-GEMM kernel (vendored verbatim). **NOTE: a latent int32 base-pointer overflow (>~1.05M rows) — NOT patched here; fix (int64) + re-verify in ex09, then re-sync. The separate `TP=1` fused crash (at realistic M ≤ 1.05M) is still OPEN — see the MoE section.**
- `[contract RT1/RT2/RT5]` → ex05 routing-partition contracts on `offsets` (now over the **local** partition under EP).

```sh
grep -rn "\[C4\]\|\[C5-b\]\|\[C6\]\|\[C7/C8\]\|\[C9\]\|\[contract" nanovllm/
```

## Scope / status

- **Single-GPU and expert-parallel** with a **TP/EP device mesh** (`world = TP × DP`,
  step 2) and two EP schedules — **C6** (`all_reduce`, default) and the **C7/C8 hybrid
  block** (`MOE_EP_MODE=hybrid`). `tp_size < world` is expressible: **C8** shape
  (`TP=4 DP=2`), **C7** shape (`TP=1 DP=8`). `DP=1` is byte-identical to the old flat
  path. Still **no DP engine** (step 3) — `DP>1` replicas process the *replicated* batch
  redundantly until the batch is partitioned. `ep ∈ {1,2,4,8}` (128 % ep == 0).
- **Validated quantitatively** against the production-vLLM GSM8K oracle (strict
  **0.8923**): C6 lands **0.8886–0.8969 across ep=1–8** (loop and fused, within the
  oracle's ±0.85%). The **mesh + hybrid** reproduce it across shapes: DP=1 hybrid
  **0.8939** (tp=8 fused) / allreduce **0.8954**; **C8** `TP=4 DP=2` hybrid **0.8923**
  (fused); **C7** `TP=1 DP=8` hybrid **0.8908** (loop) — so the device mesh and
  dispatch/combine are correct at every tp. **Caveat:** `TP=1` with `MOE_KERNEL=fused`
  crashes (CUDA illegal memory access) — an **OPEN** C9 kernel bug (see above); `loop`
  is the workaround and gives the correct 0.8908.
- **Speed:** fused ≫ loop; EP throughput peaks around **ep=4** for this
  model+batch (ep=8 is past the knee — the `all_reduce` + expert-load-imbalance
  straggler grow faster than per-rank compute shrinks). vLLM stays ~2–3× faster on
  generation (mature engine, tuned/graph-able MoE kernel); closing that is future
  work, and the kernel is a swappable verified component.
```
