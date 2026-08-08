# Exercise 3 — Multi-Head Attention under TP

**Goal**: TP-shard a full MHA block (Q, K, V, O + RoPE + SDPA).

## The key idea

Sharding attention is natural because attention is **head-parallel**: each
head's math is independent of the others. Under TP-N you give each rank
`H / N` heads and it computes attention only for those heads. The final
`o_proj` collapses them back with a row-parallel all-reduce.

The one non-obvious piece: Q, K, V all shard identically (same heads), so we
fuse them into ONE packed projection to save GEMM launches. The weight
layout on the output dim is `[Q_heads | K_heads | V_heads]`, and each rank
takes a contiguous slice of each of the three groups.

```
full weight (out-dim):   [Q_0 Q_1 ... Q_{H-1} | K_0 ... K_{H-1} | V_0 ... V_{H-1}]
tp_size=4, rank=1 slice: [Q_{H/4} ... Q_{2H/4-1} | K_{H/4} ... | V_{H/4} ...]
```

## What to fill in

1. `QKVParallelLinear.weight_loader(full_weight, shard_id)` — the offset math
   for "q" vs "k" vs "v". Follow the docstring's step list.
2. `TPMHA.__init__` — instantiate QKVParallelLinear + RowParallelLinear.
3. `TPMHA.forward` — the nine-step recipe in the docstring. The tricky part is
   the split: don't do `qkv.chunk(3, dim=-1)` since the local slice sizes for
   Q, K, V are all equal for MHA but you should use `.split([...])` with the
   explicit sizes so ex04 can extend without rewriting.

## Numerical gotcha

SDPA in bf16 accumulates in fp32 internally, but the input dtype matters for
tolerance. With H=8 heads and D=32, expect bf16 to match within ~5%. The
test uses the conftest tolerance table (`tol(dtype)`).

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest -x bootcamp/tests/test_ex03_mha_tp.py -v
```

## What ex04 will add

Exercise 4 will:
- Loosen the `num_heads == num_kv_heads` assertion (GQA).
- Handle `tp_size > num_kv_heads` by *replicating* KV shards across ranks —
  the fix that's missing from `nanovllm-jun/nanovllm/layers/linear.py`.

Get this exercise clean first; ex04 subclasses this one.
