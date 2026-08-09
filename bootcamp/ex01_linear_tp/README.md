# Exercise 1 — Linear Tensor Parallelism

**Goal**: implement `ColumnParallelLinear` and `RowParallelLinear`, the two
building blocks that make TP work for any subsequent MLP / attention layer.

## The math you're implementing

A dense linear `y = x @ W^T` (weight `W` has shape `[out, in]`) can be split
two ways:

**Column parallel — shard the output dim.** Weight becomes
`W_local` of shape `[out/N, in]` on each of N ranks. Each rank computes
`y_local = x @ W_local^T` of shape `[..., out/N]`. No communication yet — the
output is *sharded*. Downstream logic decides whether to keep it sharded
(preferred: feed straight into a RowParallelLinear) or gather it.

**Row parallel — shard the input dim.** Weight becomes `W_local` of shape
`[out, in/N]`. Each rank consumes its input shard `x_local` of shape
`[..., in/N]`, computes a *partial* `y_partial = x_local @ W_local^T` of
shape `[..., out]`, and the ranks sum these partials via `all_reduce(SUM)`
to get the full `y`.

Composing them: `ColumnParallelLinear → RowParallelLinear` produces a full
replicated output with **one** collective (the row's all-reduce). That's why
MLPs are TP'd as `col → row`, not the reverse.

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
