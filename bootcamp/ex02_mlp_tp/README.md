# Exercise 2 — SwiGLU MLP under TP

**Goal**: use your ex01 `ColumnParallelLinear` + `RowParallelLinear` to build
a full SwiGLU MLP, and along the way add the "merged" variant of column
parallelism that fuses `gate` and `up` into one weight matrix.

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
