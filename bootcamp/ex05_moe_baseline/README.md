# Exercise 5 — MoE baseline (single-GPU, two layouts)

**Goal**: build the MoE forward pass from scratch, in two flavors:

- **Ex05a — Naive**: per-expert Python loop, direct scatter/gather.
  The "MoE hello world."
- **Ex05b — Permuted**: same math, permuted layout. Each expert
  processes a contiguous slice. Bridges directly to Ex06 (EP) and
  Ex08 (fused Triton kernel).

Both are single-GPU, no parallelism. This exercise focuses on the
**sparse routing math** and the **permutation vocabulary** — the two
skills you need before Ex06 sprinkles all-to-all comm on top.

## The math

Given input tokens $X \in \mathbb{R}^{N \times H}$, a router weight
$W_g \in \mathbb{R}^{E \times H}$ (E = num_experts), and E expert
SwiGLU MLPs $\{f_e\}_{e=0}^{E-1}$ each with weights
$W_g^{(e)}, W_u^{(e)}, W_d^{(e)}$.

**Routing** (softmax over experts, then top-k):

$$
G \;=\; \mathrm{softmax}(X W_g^\top) \;\in\; \mathbb{R}^{N \times E}
$$

Take the top-k values per row:

$$
(w_{ij}, e_{ij})_{j=1..k} \;=\; \mathrm{topk}(G_i, k) \quad \text{for each token } i.
$$

If `norm_topk_prob=True`, renormalize:

$$
w_{ij} \gets w_{ij} \Big/ \sum_{j'} w_{ij'}
$$

**Combine** (each token's output is the weighted sum of its k experts' outputs):

$$
Y_i \;=\; \sum_{j=1}^{k} w_{ij} \, f_{e_{ij}}(X_i)
$$

## Ex05a — Naive per-expert loop

Directly iterate over experts, gather each expert's assigned tokens,
compute, scatter-add:

```python
Y = zeros_like(X)
for e in range(num_experts):
    token_idx, k_idx = where(selected_experts == e)      # tokens routed to expert e
    if token_idx.numel() == 0: continue
    Y_e = f_e(X[token_idx])
    Y.index_add_(0, token_idx, Y_e * routing_weights[token_idx, k_idx, None])
```

Simple, correct, obvious. But each expert typically gets a small number
of tokens (`N × top_k / E` on average), and each `f_e` call is a tiny
matmul — poor GEMM efficiency, high kernel-launch overhead per expert.
This is the pattern in HF's `Qwen3MoeSparseMoeBlock` and the same one
the intern's nanovllm-jun implementation uses.

## Ex05b — Permuted with grouped compute

Reorganize the data first so each expert's inputs are contiguous, then
run one grouped matmul per expert on the contiguous slice:

```
Step 1: Router (same as naive)                              [N, top_k] weights + expert IDs
Step 2: Flatten routing into N*top_k records                (token, expert, weight) triples
Step 3: argsort by expert → permutation                     [N*top_k] permutation indices
Step 4: bincount + cumsum → offsets                         [num_experts + 1]
Step 5: gather sorted input                                 sorted_x = x[sorted_token_ids]
Step 6: grouped compute
        for e in range(num_experts):
            sorted_out[offsets[e]:offsets[e+1]] = f_e(sorted_x[offsets[e]:offsets[e+1]])
Step 7: weight + unpermute + scatter-add                    output.index_add_(0, sorted_token_ids, sorted_out * sorted_weights[:, None])
```

Same MoE math; different data path. Each expert's `f_e` call operates
on a contiguous block of tokens — **better GEMM efficiency, better
tensor-core utilization**.

## The permutation vocabulary you're building

These five operations reappear everywhere in production MoE code and
in the paper's Verus spec:

| Operation | Shape | Meaning |
|---|---|---|
| `torch.argsort(expert_ids)` | `[N * top_k]` | Permutation that groups records by expert |
| `torch.bincount(sorted_expert_ids, minlength=E)` | `[E]` | Count per expert |
| `counts.cumsum(0)` (prepended with 0) | `[E + 1]` | Per-expert start/end offsets |
| `x[sorted_token_ids]` | `[N * top_k, H]` | Gather input in permuted order |
| `output.index_add_(0, sorted_token_ids, sorted_out)` | `[N, H]` | Unpermute + accumulate top_k contributions |

Ex06 (EP) inserts `all_to_all_variable` between steps 5 and 6 (or 4
and 5, depending on which framing) — same vocabulary, plus one
collective. Ex08 (fused MoE kernel) collapses steps 5-7 into a single
Triton kernel launch — same vocabulary, plus GPU-side grouped GEMM.

## Lifecycle — when each method runs

For both `NaiveSparseMoE` and `PermutedSparseMoE`, the standard
4-phase `nn.Module` lifecycle applies:

- **Phase 1 (Construction)**: `__init__` allocates `self.gate` (Linear)
  and `self.experts` (ModuleList of E RefSwiGLU_MLP instances). No TP
  sharding — this is single-GPU code.
- **Phase 2 (Placement)**: `.to(device, dtype)` moves everything.
- **Phase 3 (Loading)**: for testing, `self.gate.weight` and each
  expert's weights are loaded from a reference module by copying
  parameter `.data`. In production, safetensors keys look like
  `model.layers.5.mlp.gate.weight`, `model.layers.5.mlp.experts.0.gate_proj.weight`,
  etc. — a standard checkpoint walker fills them.
- **Phase 4 (Inference)**: `forward(x)` runs the routing + expert compute.

## Numerical tolerance

Ex05a's output should match `RefSparseMoE` up to fp reduction-order
noise (both accumulate in the same order, so fp32 should be
byte-close and bf16 within ~1e-2).

Ex05b's output should also match `RefSparseMoE`, but with **slightly
looser tolerance** because the accumulation order differs (permuted
vs by-expert). Expect ~1e-5 in fp32, ~5e-2 in bf16. Tests use
`assert_close` with generous atol/rtol.

## Config used in tests

- `HIDDEN = 128`
- `INTERMEDIATE = 64` (each expert's MLP intermediate dim)
- `NUM_EXPERTS = 8`
- `TOP_K = 2`
- `N_TOKENS = 32`
- `dtype ∈ {fp32, bf16}`

At Qwen3-30B-A3B scale, `num_experts = 128`, `top_k = 8`, but the
algorithm is identical — just larger constants.

## Run

```sh
CUDA_VISIBLE_DEVICES=1 uv run pytest bootcamp/tests/test_ex05a_naive_moe.py -v
CUDA_VISIBLE_DEVICES=1 uv run pytest bootcamp/tests/test_ex05b_permuted_moe.py -v
```

Both are single-GPU tests — no `run_on_ranks`, no NCCL setup. Fast
iteration.

## What Ex06 will add

Ex06 turns `PermutedSparseMoE` into `EPSparseMoE` by:

1. **Sharding the experts**: each rank owns `num_experts / ep_size`
   experts. So each rank's `self.experts` has `num_experts / ep_size`
   MLPs.
2. **Adding `all_to_all_variable` (dispatch)**: after Step 3
   (permutation), send each token to the rank that owns its assigned
   expert. Each rank ends up with tokens for its local experts only.
3. **Local grouped compute** (Step 6, but only over local experts).
4. **Adding `all_to_all_variable` (combine)**: send expert outputs
   back to their original owners.
5. **Unpermute + combine** on the receiving side.

The core algorithm and vocabulary from Ex05b transfer directly. The
new element is the two all-to-all collectives around the grouped
compute step.

## What Ex08 will add

Ex08 replaces Step 6 (the Python for-loop over experts) with a single
Triton kernel that does the grouped matmul across all experts. Same
permutation, same offsets, same unpermute — just one CUDA kernel
launch instead of `num_experts` PyTorch calls. Much better GPU
utilization.

## Traps to watch for

- **`argsort` is not stable by default in PyTorch** (until torch
  2.4+). If you rely on preserving relative order within an expert's
  bucket for numerical reproducibility, pass `stable=True`. Not
  test-failing for us; just note the flag.
- **`index_add_` is not deterministic on CUDA by default** — floating
  point associativity issues when multiple threads accumulate into the
  same output slot. For a workshop paper on verification, this is a
  known deterministic-inference concern; `torch.use_deterministic_algorithms(True)`
  can help but slows things down. For Ex05, the test tolerance handles
  the fp noise.
- **`bincount(minlength=num_experts)`** is essential — without
  `minlength`, if the last few experts get zero tokens, `bincount`
  returns a shorter tensor and your offsets are wrong. Always pass
  `minlength`.
- **`counts.cumsum(0)` returns `[E]`, not `[E+1]`.** You need to
  prepend a zero: `torch.cat([torch.zeros(1), counts.cumsum(0)])`
  gives `[E+1]` offsets where `offsets[e]` = start of expert e's
  slice.
