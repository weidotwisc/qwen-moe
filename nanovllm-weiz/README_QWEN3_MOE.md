# Qwen3-30B-A3B (Qwen3-MoE) support for nano-vLLM

This fork adds **Qwen3 Mixture-of-Experts** inference (e.g. Qwen3-30B-A3B) to
nano-vLLM, single-GPU.

**Base:** clean checkout of upstream nano-vLLM
(`github.com/GeeeekExplorer/nano-vllm`) at commit
**`bb823b3e06983d71485a8e1f23715ebd87d98ef8`**
("Merge pull request #218 from GeeeekExplorer/chunked-prefill-refactor",
2026-04-26). That pristine base is committed on its own (qwen-moe commit
`aba5ea76`) so this feature is a clean diff on top of it —
`git diff aba5ea76 HEAD` is the exact delta.

Stock nano-vLLM supports **dense** Qwen3 only. The sole architectural delta for
Qwen3-MoE is the FFN (dense MLP → sparse MoE); everything else — GQA attention
with QK-norm, RoPE, RMSNorm, embeddings — is reused from `models/qwen3.py`.

## What changed

| File | Change |
|---|---|
| `nanovllm/models/qwen3_moe.py` | **new** — `Qwen3MoeForCausalLM` + `Qwen3MoeSparseMoeBlock`; reuses dense `Qwen3Attention`. |
| `nanovllm/layers/fused_moe.py` | **new** — fused MoE Triton grouped-GEMM (vendored from `bootcamp/ex09_fused_moe`). |
| `nanovllm/utils/loader.py` | expert-weight routing: `experts.{e}.{gate,up,down}_proj` → stacked param slice `[e]`. |
| `nanovllm/engine/model_runner.py` | dispatch the model class on `hf_config.architectures[0]`. |
| `smoke_moe.py` | **new** — single-GPU coherence smoke test. |

## MoE block: two expert kernels on one representation

Experts are stored **stacked** (`w_gate/w_up/w_down`, shapes
`[E,I,H] / [E,I,H] / [E,H,I]`) — the single layout shared by both compute paths,
selected by the `MOE_KERNEL` env var:

- `MOE_KERNEL=loop` (default) — per-expert grouped compute (reference path).
- `MOE_KERNEL=fused` — fused Triton grouped-GEMM (~2.5× faster in the smoke test).

Routing is identical either way: router → top-k → renormalize → sort by expert →
offset array → grouped compute → weight + unpermute + scatter-combine.

## Run (single GPU)

Use an environment with torch 2.11 + triton 3.6 + transformers (the repo's uv
`.venv`). From this directory:

```sh
CUDA_VISIBLE_DEVICES=0 python smoke_moe.py                    # loop (default)
CUDA_VISIBLE_DEVICES=0 MOE_KERNEL=fused python smoke_moe.py   # fused
```

`enforce_eager=True` is required (the MoE forward has host syncs and
data-dependent block shapes that CUDA-graph capture cannot handle); `smoke_moe.py`
already sets it.

## Traceability

The integration is annotated with greppable tags mapping each part back to the
verified bootcamp components (for the "generated code → contracts" thread):

- `[C5-b]` → `bootcamp/ex05_moe_baseline/reference_b.py` — sorted-offset routing + the loop expert path.
- `[C9]` → `bootcamp/ex09_fused_moe/solution.py` — the fused grouped-GEMM kernel.
- `[contract RT1/RT2/RT5]` → ex05 routing-partition contracts on `offsets`
  (`o[0]=0`, `o[E]=T·top_k`, monotone, per-expert block size == true count).

```sh
grep -rn "\[C5-b\]\|\[C9\]\|\[contract" nanovllm/     # locate every mapped point
```

## Scope / status

- **Single-GPU only** (tp=1, ep=1). Expert/tensor parallelism (bootcamp C6–C8) is
  not yet wired; the stacked-expert layout is the intended substrate for it.
- Validated so far by a **coherence smoke test** — both kernels produce sane
  output and agree in substance. **Quantitative** validation (GSM8K vs the
  production-vLLM oracle, ≈0.89 strict-match) is pending an eval harness.
