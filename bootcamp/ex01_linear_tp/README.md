# Exercise 1 — Linear Tensor Parallelism

**Goal**: implement `ColumnParallelLinear` and `RowParallelLinear`, the two
building blocks that make TP work for any subsequent MLP / attention layer.

## The math

A dense linear layer, in PyTorch's storage convention:

$$
y \;=\; x\, W^{\top}, \qquad
W \in \mathbb{R}^{O \times I},\;
x \in \mathbb{R}^{B \times I},\;
y \in \mathbb{R}^{B \times O}.
$$

where $B$ is the batch (or seq) axis, $I$ is `in_features`, and $O$ is
`out_features`. This is the operation both TP variants factor.

### ColumnParallelLinear — shard the output dim

Decompose $W$ into $N$ **row blocks** (each is a chunk along dim 0):

$$
W \;=\; \begin{bmatrix} W^{(0)} \\[2pt] W^{(1)} \\[2pt] \vdots \\[2pt] W^{(N-1)} \end{bmatrix},
\qquad W^{(r)} \in \mathbb{R}^{(O/N) \times I}.
$$

Each rank $r$ owns $W^{(r)}$ and computes

$$
y^{(r)} \;=\; x\, (W^{(r)})^{\top} \;\in\; \mathbb{R}^{B \times (O/N)}.
$$

The full $y$ is the horizontal concatenation of shards
$y = [\, y^{(0)} \mid y^{(1)} \mid \cdots \mid y^{(N-1)} \,]$ — but **you
never form $y$ explicitly**. Each rank keeps its shard and either feeds it
into a `RowParallelLinear` next (preferred, no gather) or explicitly
all-gathers if downstream code needs the full tensor. **Zero collectives**
inside `forward`.

### RowParallelLinear — shard the input dim

Decompose $W$ into $N$ **column blocks** (each is a chunk along dim 1):

$$
W \;=\; \bigl[\, W^{(0)} \mid W^{(1)} \mid \cdots \mid W^{(N-1)} \,\bigr],
\qquad W^{(r)} \in \mathbb{R}^{O \times (I/N)}.
$$

Assume the input is already sharded to match:
$x = [\, x^{(0)} \mid \cdots \mid x^{(N-1)} \,]$ with
$x^{(r)} \in \mathbb{R}^{B \times (I/N)}$. Then

$$
y \;=\; x\, W^{\top} \;=\; \sum_{r=0}^{N-1} x^{(r)}\, (W^{(r)})^{\top}.
$$

Each rank $r$ computes its partial
$y^{(r)}_{\text{partial}} = x^{(r)} (W^{(r)})^{\top} \in \mathbb{R}^{B \times O}$
(full width, wrong values), and **one** `all_reduce(SUM)` sums the partials
to the true $y$:

$$
y \;=\; \sum_r y^{(r)}_{\text{partial}}.
$$

### Composed pattern: Column → Row

`ColumnParallelLinear` outputs $y \in \mathbb{R}^{B \times (O/N)}$ — sharded
on the last axis. That's **exactly the shape `RowParallelLinear` expects as
its input** (where its own $I' = O$, split $N$ ways). So the two compose
directly, with only the Row's all-reduce as the single collective for the
whole pair. That's why an MLP is TP'd as `col → row` and never the reverse:
reversing would need an intermediate all-gather.

### Mapping to code

| Math | In your `solution.py` |
|---|---|
| $W^{(r)} \in \mathbb{R}^{(O/N) \times I}$ | `ColumnParallelLinear.__init__` allocates `nn.Parameter(torch.empty(O//N, I))` |
| $W^{(r)} \in \mathbb{R}^{O \times (I/N)}$ | `RowParallelLinear.__init__` allocates `nn.Parameter(torch.empty(O, I//N))` |
| $r$-th row-block slice | `ColumnParallelLinear.weight_loader` copies `full[r*O/N : (r+1)*O/N, :]` |
| $r$-th column-block slice | `RowParallelLinear.weight_loader` copies `full[:, r*I/N : (r+1)*I/N]` |
| $x (W^{(r)})^{\top}$ | `F.linear(x, self.weight)` in both classes |
| all-reduce SUM | `dist.all_reduce(y, group=self.group)` in `RowParallelLinear.forward` only |

## Lifecycle — when each method is called

None of the methods you're implementing call each other directly. Each is
invoked at a specific phase in the module's lifetime by external code
(the test, or in production a checkpoint loader). Timeline:

```python
# ────────────── Phase 1: Construction ──────────────
layer = ColumnParallelLinear(H, O, tp_size, tp_rank, group=None)
# ↓ __init__ runs, allocates self.weight = nn.Parameter(torch.empty(O/N, H))
#
# state: Parameter allocated with UNINITIALIZED data,
#        on the default device (CPU) + dtype (fp32).

# ────────────── Phase 2: Placement ──────────────
layer.to(device="cuda", dtype=torch.bfloat16)
# ↓ nn.Module.to() walks all Parameters and re-materializes them.
#
# state: Parameter now on cuda/bf16, still holding garbage bytes.

# ────────────── Phase 3: Weight loading ──────────────
# EXTERNALLY called — not by the module itself:
layer.weight_loader(ref.proj.weight.data)
#
# state: this rank's Parameter now holds the correct rank-local shard.
#        Same nn.Parameter Python object as after __init__; only .data changed.

# ────────────── Phase 4: Inference ──────────────
y = layer(x)               # x on cuda/bf16, shape [B, H]
# ↓ forward runs (F.linear + optional all_reduce for RowParallel).
#
# state: y is sharded (Column) or replicated (Row). Parameters unchanged.
# Phase 4 can repeat many times per Phase 3.
```

### Key invariants across the phases

- **Phase 1 must produce the Parameter.** If you defer allocation to
  `weight_loader`, Phase 2's `.to()` has nothing to move — silent bug
  where the Parameter later inherits the device/dtype of whatever
  tensor got passed to `weight_loader`.
- **Phases 2 and 3 commute** — you can `.to()` before or after loading.
- **Phase 3 is called externally.** The module never invokes its own
  `weight_loader`. The test / checkpoint loader does. Search the module
  code for calls to `.weight_loader(` and you'll find none — expected.
- **Phase 4 doesn't modify Parameters.** Forward only reads them.

### Where each of your implementations sits in the timeline

| Method | Called during phase | Called by |
|---|---|---|
| `ColumnParallelLinear.__init__` | 1 (Construction) | test / user code |
| `RowParallelLinear.__init__` | 1 (Construction) | test / user code |
| `ColumnParallelLinear.weight_loader` | 3 (Loading) | test / checkpoint loader |
| `RowParallelLinear.weight_loader` | 3 (Loading) | test / checkpoint loader |
| `ColumnParallelLinear.forward` | 4 (Inference) | test / user code |
| `RowParallelLinear.forward` | 4 (Inference) | test / user code |

In production, Phase 3 is driven by a walker like
[nanovllm-jun/utils/loader.py](../../nanovllm-jun/nanovllm/utils/loader.py)
that iterates `safetensors` file keys and calls each Parameter's registered
`weight_loader` in turn. The bootcamp tests skip that machinery and call
`weight_loader` directly — but the shape of the API is the same.

## What to fill in

Both classes have three stubs:
1. `__init__` — assert divisibility, allocate the shard-sized `nn.Parameter`.
2. `weight_loader` — take a full `[out, in]` tensor, pick this rank's slice,
   copy into `self.weight.data`. Use `narrow` or `chunk`.
3. `forward` — an `F.linear` call; the row variant additionally issues one
   `dist.all_reduce`.

Look at [solution.py](solution.py) for the exact signatures and the
`TODO(you)` comments.

## How to run the tests

```sh
# on the qwen-moe root, .venv activated (or use `uv run`)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest -x tests/test_ex01_linear_tp.py -v
```

The test parametrizes `tp_size ∈ {1, 2, 4, 8}` and `dtype ∈ {fp32, bf16}`.
`-x` fails fast on the first bad case. Green pytest = ex01 done.

## Common bugs to watch for

- **Shape math**: it's `[out, in]` in PyTorch's `nn.Linear.weight`, not
  `[in, out]`. Easy to get transposed.
- **Copy semantics**: `self.weight.data.copy_(shard)` vs
  `self.weight = nn.Parameter(shard)`. Use the former — you already allocated
  the parameter in `__init__` and mutating `self.weight` will break the
  gradient graph and any external optimizer state.
- **Contiguity**: after `narrow`, the result is a view — that's fine for
  `copy_`, but if you need to feed it to a matmul you'd want `.contiguous()`.
  (Not an issue in this exercise, but a lurking one later.)
- **all_reduce is in-place**: `dist.all_reduce(y)` mutates `y` and returns
  None. Don't do `y = dist.all_reduce(y)`.
