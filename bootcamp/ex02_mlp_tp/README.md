# Exercise 2 — SwiGLU MLP under TP

**Goal**: use your ex01 `ColumnParallelLinear` + `RowParallelLinear` to build
a full SwiGLU MLP, and along the way add the "merged" variant of column
parallelism that fuses `gate` and `up` into one weight matrix.

## The math

Let $X \in \mathbb{R}^{B \times H}$ be the batch of inputs (batch size $B$,
hidden dim $H$), $I$ the intermediate dim, and

$$
W_{g}, W_{u} \in \mathbb{R}^{I \times H}, \qquad W_{d} \in \mathbb{R}^{H \times I}.
$$

The SwiGLU MLP computes

$$
Y \;=\; \bigl[\, \mathrm{SiLU}(X W_g^{\top}) \;\odot\; (X W_u^{\top}) \,\bigr]\, W_d^{\top}\;\in\;\mathbb{R}^{B \times H},
$$

where $\odot$ is elementwise multiply and $\mathrm{SiLU}(z) = z \cdot \sigma(z)$.

### Merged formulation

Stack $W_g$ and $W_u$ vertically to fuse the two projections into one GEMM:

$$
W_{gu} \;=\; \begin{bmatrix} W_g \\[2pt] W_u \end{bmatrix} \in \mathbb{R}^{2I \times H}, \qquad
[G \mid U] \;=\; X\, W_{gu}^{\top} \;\in\; \mathbb{R}^{B \times 2I},
$$

with $G, U \in \mathbb{R}^{B \times I}$. Then $Y = \bigl[\mathrm{SiLU}(G) \odot U\bigr] W_d^{\top}$.
One `F.linear` call replaces two. The math is identical.

### Under TP with $N$ ranks

Each rank $r \in \{0, \dots, N-1\}$ holds shards of the three weights:

$$
W_{gu}^{(r)} \in \mathbb{R}^{(2I/N) \times H}
\quad(\text{rows of } W_{gu}, \text{i.e.}\ [\text{gate}_r ; \text{up}_r]),
$$

$$
W_d^{(r)} \in \mathbb{R}^{H \times (I/N)}
\quad(\text{columns of } W_d, \text{i.e.}\ W_d[:,\, r I / N : (r{+}1) I / N]).
$$

The per-rank forward is:

$$
[G_r \mid U_r] \;=\; X\, (W_{gu}^{(r)})^{\top} \;\in\; \mathbb{R}^{B \times (2I/N)},
$$

$$
Z_r \;=\; \mathrm{SiLU}(G_r) \odot U_r \;\in\; \mathbb{R}^{B \times (I/N)}
\quad(\text{elementwise, no comm needed}),
$$

$$
Y_r \;=\; Z_r\, (W_d^{(r)})^{\top} \;\in\; \mathbb{R}^{B \times H}
\quad(\text{partial output on this rank}),
$$

$$
Y \;=\; \sum_{r=0}^{N-1} Y_r
\quad(\text{one \texttt{all\_reduce}, restores the full output}).
$$

Exactness follows from $Z\, W_d^{\top} = \sum_r Z_r\, W_d^{(r)\top}$ where
$Z = [Z_0 \mid \cdots \mid Z_{N-1}]$ is the horizontal concatenation of the
rank-local slices. **The SiLU-and-multiply preserves the sharded layout**,
so no communication is needed between the merged column projection and the
row projection — that's the entire reason (Column → SiLU⊙ → Row) is the
canonical MLP TP pattern.

### Mapping to code

| Math | Class in your solution |
|---|---|
| $W_{gu}^{(r)}$ storage + weight_loader (per shard_id) | `MergedColumnParallelLinear` |
| $X (W_{gu}^{(r)})^{\top}$ | `MergedColumnParallelLinear.forward` (via `F.linear`) |
| Chunk into $[G_r \mid U_r]$, then $\mathrm{SiLU}(G_r) \odot U_r$ | `TPSwiGLUMLP.forward` middle lines |
| $Z_r (W_d^{(r)})^{\top}$ + `all_reduce` | `RowParallelLinear.forward` (already in ex01) |

## Why merged?

In Llama/Qwen/etc. the MLP has two column-parallel projections that share the
same input `x`:

```
gate = x @ W_gate^T
up   = x @ W_up^T
out  = SiLU(gate) * up
```

If you TP-shard `W_gate` and `W_up` as *separate* `ColumnParallelLinear`s,
you get two shard-storage buffers and two GEMM calls per forward. Merging
them into `[W_gate ; W_up]` (concatenated on the out-dim before sharding)
lets you:

- Fire **one** GEMM instead of two.
- Match the packed-shape convention that safetensors typically stores.

The math is exactly the same — you just chunk the output back into `(gate, up)`
before the SiLU.

## Lifecycle — when each method is called

None of the methods you're implementing call each other directly. They're
each invoked at a specific phase in the module's lifetime by external code
(the test, or in production a checkpoint loader). Here's the timeline:

```python
# ────────────────────────── Phase 1: Construction ──────────────────────────
layer = TPSwiGLUMLP(H, I, tp_size, tp_rank, group=None)
# ↓ TPSwiGLUMLP.__init__ runs, which internally invokes:
#     - MergedColumnParallelLinear.__init__(hidden=H, output_sizes=[I, I], ...)
#         ↓ ColumnParallelLinear.__init__ (via super().__init__)
#             - allocates self.weight = nn.Parameter(torch.empty(2*I/N, H))
#     - RowParallelLinear.__init__(intermediate=I, hidden=H, ...)
#         - allocates self.weight = nn.Parameter(torch.empty(H, I/N))
#
# state: all Parameters allocated with UNINITIALIZED data.
#        Living on the default device (usually CPU) and dtype (usually fp32).
#        Any call to `list(layer.parameters())` returns them.

# ────────────────────────── Phase 2: Placement ────────────────────────────
layer.to(device="cuda", dtype=torch.bfloat16)
# ↓ nn.Module.to() walks all Parameters recursively and re-materializes them
#   on the target device/dtype.
#
# state: Parameters still hold garbage bytes, but now on cuda:rank in bf16.

# ────────────────────────── Phase 3: Weight loading ───────────────────────
# Externally called — NOT by the module itself. In test:
layer.gate_up_proj.weight_loader(ref.gate_proj.weight.data, 0)   # fills [0 : I/N] of merged buffer
layer.gate_up_proj.weight_loader(ref.up_proj.weight.data,   1)   # fills [I/N : 2I/N] of merged buffer
layer.down_proj.weight_loader(ref.down_proj.weight.data)         # fills the whole down buffer
#
# In production, a top-level walker (like nanovllm-jun/utils/loader.py)
# iterates over safetensors keys and dispatches to the right weight_loader.
#
# state: this rank's Parameters now hold correct rank-local shard values.
#        Same Parameter Python objects as after __init__ (only .data changed).

# ────────────────────────── Phase 4: Inference ────────────────────────────
y = layer(x)             # x on cuda/bf16, shape [B, H], replicated across ranks
# ↓ TPSwiGLUMLP.forward runs, which invokes (in order):
#     - self.gate_up_proj.forward(x)     → F.linear + returns sharded gate_up
#     - .chunk(2, dim=-1) → (gate, up)
#     - F.silu(gate) * up                → elementwise on the shard
#     - self.down_proj.forward(hidden)   → F.linear + all_reduce(SUM)
#
# state: y is replicated across ranks. Parameters unchanged.
# Phase 4 can repeat many times per Phase 3.
```

### Key invariants across the phases

- **Phase 1 must produce every Parameter.** If you defer parameter
  allocation to `weight_loader`, Phase 2's `.to()` has nothing to move
  (silent bug — Parameter later lands wherever the input tensor happens
  to be, which may or may not match).
- **Phases 2 and 3 commute** (you can `.to()` before or after loading;
  they touch the same buffers on the same rank).
- **Phase 3 is idempotent per shard_id.** Calling `weight_loader(w, 0)`
  twice with different `w` is legal — the second call overwrites the
  gate slice, up remains untouched. This matters for hot-reload scenarios
  (rare in inference; common in fine-tuning).
- **Phase 4 does not modify Parameters** — it reads them, produces `y`,
  and can be called any number of times.
- **`weight_loader` is external.** The module never invokes its own
  `weight_loader`. The test / checkpoint loader does. Search the module
  code for calls to `.weight_loader(` and you'll find none — that's
  the correct answer, not a bug.

### Where each of your implementations lives in the timeline

| Method | Called during phase | Called by |
|---|---|---|
| `MergedColumnParallelLinear.__init__` | 1 (Construction) | `TPSwiGLUMLP.__init__` (via `super()`) |
| `TPSwiGLUMLP.__init__` | 1 (Construction) | test / user code |
| `MergedColumnParallelLinear.weight_loader` | 3 (Loading) | test / checkpoint loader — **twice** per instance (gate + up) |
| `RowParallelLinear.weight_loader` | 3 (Loading) | same — once per instance |
| `TPSwiGLUMLP.forward` | 4 (Inference) | test / user code, potentially many times |
| `MergedColumnParallelLinear.forward` | 4 (Inference) | `TPSwiGLUMLP.forward` |
| `RowParallelLinear.forward` | 4 (Inference) | `TPSwiGLUMLP.forward` |

## What to fill in

1. `MergedColumnParallelLinear.weight_loader(full_weight, shard_id)` — the
   only change from ex01's column loader is picking the right slice within
   `self.weight` based on `shard_id`.
2. `TPSwiGLUMLP.__init__` — one `MergedColumnParallelLinear`, one
   `RowParallelLinear`. Bias always False.
3. `TPSwiGLUMLP.forward` — the four-line dance in the docstring.

## Collective budget

**One** `all_reduce` per forward, from the `RowParallelLinear` at the end.
If you're issuing more, something's off.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest -x bootcamp/tests/test_ex02_mlp_tp.py -v
```
