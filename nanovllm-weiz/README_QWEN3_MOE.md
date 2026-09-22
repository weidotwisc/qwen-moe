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
| `nanovllm/layers/fused_moe.py` | **new** — fused MoE Triton grouped-GEMM (vendored **verbatim** from `bootcamp/ex09_fused_moe`). NOT the cause of the tp=1 crash (that was KV-cache sizing — see below); it does have a *latent* int32 overflow at >~1.05M rows, documented, **not patched here** (fix + re-verify in ex09, re-sync). |
| `nanovllm/utils/loader.py` | expert-weight routing: parses the global expert id and dispatches to the block's stacked param loader (the per-rank EP shard-map/skip lives in `qwen3_moe.py`). |
| `nanovllm/layers/linear.py` | **[C4]** `QKVParallelLinear` — GQA + **KV-head replication** when `tp_size > num_kv_heads` (sizing + replication-aware weight loading). |
| `nanovllm/models/qwen3.py` | **[C4]** `Qwen3Attention` allows `tp_size > num_kv_heads` (`num_kv_heads = max(1, …)`). |
| `nanovllm/engine/model_runner.py` | dispatch model class on `architectures[0]`; **[C4]** KV-cache sized for the per-rank (replicated) head count; env-overridable NCCL rendezvous port (`NANOVLLM_DIST_PORT`) for same-node multi-instance. |
| `smoke_moe.py` | **new** — single-GPU coherence smoke test. |
| `gsm8k_moe.py` | **new** — GSM8K accuracy harness (loop/fused × any `TP`/`DP`), the quantitative integration test. |
| `nanovllm/utils/parallel.py` | **new [C7/C8 mesh]** — TP/EP device mesh + accessors (`get_tp_*`/`get_ep_*`); `world = TP × DP = #GPUs = ep_size`. |
| `nanovllm/config.py`, `engine/llm_engine.py`, `engine/model_runner.py` | **[C7/C8 mesh]** `data_parallel_size` knob; spawn `world = TP×DP` ranks; build `tp_group`/`ep_group`; KV sized by `tp_size`; **`num_kvcache_blocks` reconciled to the global `min` across ranks** (fixes a `tp=1` KV-cache OOB — see below). |
| `nanovllm/layers/linear.py`, `layers/embed_head.py`, `models/qwen3.py` | **[C7/C8 mesh]** shard / all_reduce / LM-head gather over `tp_group` via the accessors (was the flat world). |
| `repro_fused_moe_bug.py` | **new** — single-GPU probe + fused-vs-loop check for the *latent* [C9] int32 overflow (a separate issue from the fixed tp=1 KV-cache crash). |
| `engine/llm_engine.py`, `engine/model_runner.py`, `engine/sequence.py`, `utils/parallel.py` | **[DP step 3]** per-replica `Scheduler`s + round-robin admission; `run_dp` synchronized global step + `run_dummy` lockstep filler for idle replicas; per-replica leader sampling + dp-leader `gather_object` token return; `Sequence` pickles `temperature`. `DP=1` byte-identical. |

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

`tp_size < world` is expressible: `TP=4 DP=2` is the **C8** shape (two TP groups over
one EP world), `TP=1 DP=8` is **C7** (pure EP). **[step 3] the DP engine is implemented**:
`llm_engine` holds one `Scheduler` per replica (requests round-robin'd at admission,
KV never migrates); `model_runner.run_dp` runs a synchronized global step where each
replica forwards its **distinct** batch shard; a drained/idle replica runs a **dummy
forward** (`run_dummy`, 1 token, `slot=-1`) to stay in `ep_group`-collective lockstep;
each replica's leader samples its shard and a **dp-leader `gather_object`** returns
tokens to rank 0. So `DP>1` is now a real throughput win, not redundant work. **`DP>1`
requires `MOE_EP_MODE=hybrid`** (allreduce mode all-reduces `[T,H]` over the world →
shape mismatch across replicas; asserted in `model_runner`). `ep_size==1` is the
single-GPU path.

### [C7/C8] `TP=1` KV-cache OOB — FIXED (and why it looked like a kernel bug)

`MOE_EP_MODE=hybrid` at **`TP=1 DP=8`** full-scale GSM8K crashed with a **CUDA illegal
memory access** — but *not* in the MoE. A synchronous (`CUDA_LAUNCH_BLOCKING=1`)
traceback put it in **`store_kvcache`** (the attention KV-cache write), and the cause
is **per-rank `num_kvcache_blocks` that was never reconciled**:

- Each rank sizes its KV cache from *its own* free memory after warmup
  (`model_runner.allocate_kv_cache`) — no cross-rank agreement.
- The **rank-0 scheduler** hands out block-ids from *its* count to **all** ranks. If a
  worker sized a smaller cache, a scheduled slot (`slot = block_id*block_size + …`)
  overflows that worker's cache → `store_kvcache_kernel` writes out of bounds → fault.
- At `TP=1` the warmup MoE-dispatch is lopsided (uniform warmup input → a few hot
  experts), so hot-expert ranks hit higher peak memory → fewer blocks. Observed: ranks
  6,7 = **1902** blocks, rank 0 = **2121** → rank 0 over-budgets ranks 6,7 → OOB on
  exactly ranks 6,7 (the crash ranks). Invisible under sharded TP (identical footprints).

**Fix:** `all_reduce` **MIN** of `num_kvcache_blocks` across ranks in `allocate_kv_cache`
(what vLLM does) → every rank's cache holds any block the scheduler assigns. After it,
`TP=1 DP=8` fused completes: **strict 0.9014** (in-band).

**Why `loop` looked fine (a red herring):** the KV budget derives from *free memory
after warmup*, and loop vs fused have different warmup peaks → different block counts →
loop's spread happened not to put rank 0 over a worker. "loop works" wrongly implicated
the fused kernel; it was never a kernel bug. The `store_kvcache` traceback (captured
under blocking) was the real evidence — not the loop/fused split.

*(Separately, `fused_moe.py` has a **latent** int32 base-pointer overflow at >~1.05M
rows — `repro_fused_moe_bug.py` reproduces it; unreachable at `TP=1`'s ≤1.05M rows,
kept verbatim to ex09, fix + re-verify there then re-sync.)*

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
- `[C9]`   → `bootcamp/ex09_fused_moe/solution.py` — fused grouped-GEMM kernel (vendored verbatim). **NOTE: a *latent* int32 base-pointer overflow (>~1.05M rows) — NOT patched here; fix (int64) + re-verify in ex09, then re-sync. (The `TP=1` crash was NOT this kernel — it was a KV-cache sizing bug, now FIXED; see the MoE section.)**
- `[contract RT1/RT2/RT5]` → ex05 routing-partition contracts on `offsets` (now over the **local** partition under EP).

```sh
grep -rn "\[C4\]\|\[C5-b\]\|\[C6\]\|\[C7/C8\]\|\[C9\]\|\[contract" nanovllm/
```

## Scope / status

- **Single-GPU and expert-parallel** with a **TP/EP device mesh** (`world = TP × DP`,
  step 2) + a real **data-parallel engine** (step 3) and two EP schedules — **C6**
  (`all_reduce`, default) and the **C7/C8 hybrid block** (`MOE_EP_MODE=hybrid`).
  `tp_size < world` runs true **C8** (`TP=4 DP=2`) and **C7** (`TP=1 DP=8`) with each
  replica on a **distinct** batch shard. `DP=1` is byte-identical to the old flat path.
  `ep ∈ {1,2,4,8}` (128 % ep == 0).
- **[step 3] Real DP — throughput win** (full 1319 GSM8K, 8×A100, fused hybrid): DP=1
  (TP=8) **0.9030 / 309s** → C8 (TP=4 DP=2) **0.8992 / 234s (1.3×)** → C7 (TP=1 DP=8)
  **0.8939 / 153s (2.0×)**. Monotonic wall-clock speedup at in-band accuracy — the
  payoff that step 2's redundant replicas lacked. Validated across mixed prefill/decode
  and drained-replica dummy-forward lockstep with no NCCL hang.
- **Validated quantitatively** against the production-vLLM GSM8K oracle (strict
  **0.8923**): C6 lands **0.8886–0.8969 across ep=1–8** (loop and fused, within the
  oracle's ±0.85%). The **mesh + hybrid** reproduce it across shapes: DP=1 hybrid
  **0.8939** (tp=8 fused) / allreduce **0.8954**; **C8** `TP=4 DP=2` hybrid **0.8923**
  (fused); **C7** `TP=1 DP=8` hybrid **0.8908** (loop) / **0.9014** (fused) — so the
  device mesh and dispatch/combine are correct at every tp. A `TP=1` KV-cache OOB
  (per-rank `num_kvcache_blocks` never reconciled) once crashed the fused path;
  **fixed** by global-`min` reconciliation (see above) — fused now completes at 0.9014.
- **Speed:** fused ≫ loop; EP throughput peaks around **ep=4** for this
  model+batch (ep=8 is past the knee — the `all_reduce` + expert-load-imbalance
  straggler grow faster than per-rank compute shrinks). vLLM stays ~2–3× faster on
  generation (mature engine, tuned/graph-able MoE kernel); closing that is future
  work, and the kernel is a swappable verified component.

## Changelog (development journal)

Chronological (newest last); each is a commit on `main` — `git show <hash>` for detail.

- **`aba5ea76`** — vendor pristine upstream nano-vLLM base (dense Qwen3 only).
- **`3ee1cdf`** — **[C5]** MoE baseline (loop) + **[C9]** fused Triton grouped-GEMM; stacked experts, `MOE_KERNEL=loop|fused`.
- **`fea0d6d`** — **[C6]** expert parallelism (DP=1 / one `all_reduce`) + **[C4]** GQA KV-head replication for `tp > num_kv_heads`.
- **`8307f23`** — **[C7/C8] hybrid MoE block** (`MOE_EP_MODE=hybrid`: stripe → all-to-all dispatch → combine → all_gather), run at `tp = ep = world` (DP=1) as a correctness gate.
- **`d4198ce`** — **[C7/C8 step 2] TP/EP device mesh** (`world = TP × DP`): `tp_group`/`ep_group` + accessors, so `tp < world` (C7/C8 shapes) becomes expressible. *(This commit's message wrongly blamed the `TP=1` fused crash on the C9 kernel — corrected in the next commit.)*
- **`9dbe261`** — **KV-cache OOB fix.** The `TP=1` crash was `store_kvcache` writing out of bounds: `num_kvcache_blocks` was computed per-rank from local free memory and **never reconciled**, so the rank-0 scheduler over-budgeted workers that had sized smaller caches. Fixed by `all_reduce` **MIN** across ranks (what vLLM does). It was **not** the fused kernel — that inference from "loop works" was a red herring; a *latent* int32 base-pointer overflow in the kernel (>~1.05M rows) is real but unreachable at `TP=1`, documented, left verbatim to `ex09`. Full analysis in the KV-cache section above.
- **`8734e92`** — **[DP step 3] real data-parallel engine.** Per-replica `Scheduler`s + `run_dp` synchronized global step + `run_dummy` lockstep filler for idle replicas + dp-leader `gather_object` token return; `Sequence` pickles `temperature`. `DP > 1` now runs a **distinct** batch shard per replica → **throughput win**: full 1319 GSM8K, 8×A100, fused hybrid — DP=1 (TP=8) 0.9030/309s → C8 (TP=4 DP=2) 0.8992/234s (1.3×) → C7 (TP=1 DP=8) 0.8939/153s (2.0×), in-band accuracy. `DP > 1` requires `MOE_EP_MODE=hybrid`. `DP=1` byte-identical.

**Known / deferred:** the latent [C9] int32 overflow (fix in `bootcamp/ex09` + re-sync); least-loaded DP admission (vs round-robin); CUDA-graph under DP; a Verus-verified KV-cache/scheduler control plane (this KV bug motivates it).
