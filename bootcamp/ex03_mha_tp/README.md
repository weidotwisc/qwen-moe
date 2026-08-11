# Exercise 3 — Multi-Head Attention under TP

**Goal**: TP-shard a full MHA block (Q, K, V, O + RoPE + SDPA).

## The math

Let $X \in \mathbb{R}^{B \times T \times H}$ (batch $B$, sequence $T$,
hidden $H$), with $H_q$ heads of dim $D$ (so $H_q \cdot D$ is the total
attention-projection dim; for this exercise $H_k = H_v = H_q$, i.e. pure
MHA). Weights:

$$
W_q, W_k, W_v \in \mathbb{R}^{(H_q D) \times H}, \qquad
W_o \in \mathbb{R}^{H \times (H_q D)}.
$$

The attention block (no TP) computes:

$$
\begin{aligned}
Q &= X\,W_q^{\top},\quad K = X\,W_k^{\top},\quad V = X\,W_v^{\top}
\quad \text{— each shape } [B, T, H_q D] \\
Q,K &\gets \mathrm{RoPE}(Q, K) \\
\text{per head } h: \quad \mathrm{attn}_h &= \mathrm{softmax}\!\left(\tfrac{Q_h K_h^{\top}}{\sqrt{D}}\right) V_h
\quad \text{— shape } [B, T, D] \\
\mathrm{attn\_out} &= [\,\mathrm{attn}_0 \mid \mathrm{attn}_1 \mid \cdots \mid \mathrm{attn}_{H_q - 1}\,]
\quad \text{— shape } [B, T, H_q D] \\
Y &= \mathrm{attn\_out}\, W_o^{\top}
\quad \text{— shape } [B, T, H]
\end{aligned}
$$

### Fused QKV weight (before TP sharding)

Since $W_q, W_k, W_v$ share their input $X$, stack them vertically for
one merged GEMM:

$$
W_{qkv} \;=\; \begin{bmatrix} W_q \\[2pt] W_k \\[2pt] W_v \end{bmatrix}
\in \mathbb{R}^{3 H_q D \times H},
\qquad
[Q \mid K \mid V] \;=\; X\,W_{qkv}^{\top} \in \mathbb{R}^{B \times T \times 3 H_q D}.
$$

Split with explicit sizes (**not `.chunk(3)`** — Ex04 will have unequal Q/K/V sizes):
$Q, K, V = \mathtt{qkv.split([H_q D, H_q D, H_q D], dim{=}{-}1)}$.

### Under TP with $N$ ranks

Each rank $r$ owns **$H_q / N$ heads** — a contiguous range of the head
axis. The three weights shard as follows:

**QKV (column-parallel — shard the head-out dim of the merged weight):**

$$
W_{qkv}^{(r)} \;\in\; \mathbb{R}^{3 (H_q/N) D \; \times \; H}
\quad\text{(rows: this rank's Q-heads, K-heads, V-heads stacked)}.
$$

**Attention (per-rank, no cross-rank comm):**

$$
[Q_r \mid K_r \mid V_r] \;=\; X\,(W_{qkv}^{(r)})^{\top} \;\in\; \mathbb{R}^{B \times T \times 3 (H_q/N) D}
$$

$$
\mathrm{attn}_r \;=\; \mathrm{SDPA}\bigl(\mathrm{RoPE}(Q_r), \mathrm{RoPE}(K_r), V_r\bigr)
\;\in\; \mathbb{R}^{B \times T \times (H_q/N) D}.
$$

RoPE is computed independently on each rank (the cos/sin cache is the
same across ranks — depends only on head_dim and position, not head index).

**O (row-parallel — shard the head-input dim):**

$$
W_o^{(r)} \;\in\; \mathbb{R}^{H \times (H_q/N) D}
\quad\text{(columns of } W_o \text{: this rank's head shard)}.
$$

$$
Y_r^{\text{partial}} \;=\; \mathrm{attn}_r\,(W_o^{(r)})^{\top} \;\in\; \mathbb{R}^{B \times T \times H}
\qquad Y \;=\; \sum_{r=0}^{N-1} Y_r^{\text{partial}} \;\;\text{via one all-reduce.}
$$

### Why one `all_reduce` suffices — the identity

Full $\mathrm{attn\_out} = [\,\mathrm{attn}_0 \mid \cdots \mid \mathrm{attn}_{N-1}\,]$
is the horizontal concatenation of per-rank attention outputs.
Column-block $W_o = [\,W_o^{(0)} \mid W_o^{(1)} \mid \cdots \mid W_o^{(N-1)}\,]$.
Then

$$
Y \;=\; \mathrm{attn\_out}\, W_o^{\top}
\;=\; [\,\mathrm{attn}_0 \mid \cdots \mid \mathrm{attn}_{N-1}\,]
\cdot
\begin{bmatrix} W_o^{(0)\top} \\[2pt] W_o^{(1)\top} \\[2pt] \vdots \end{bmatrix}
\;=\; \sum_{r=0}^{N-1} \mathrm{attn}_r\, (W_o^{(r)})^{\top}.
$$

**Concat-times-vstack collapses to sum-of-per-rank-products** — same
identity that makes MLP's `down_proj` row-parallel work. Row-parallel
$W_o$ + `all_reduce(SUM)` reconstructs $Y$ exactly, without ever
materializing the concatenated intermediate. Total collectives per
attention block: **one all-reduce**.

### Mapping to code

| Math | In your `solution.py` |
|---|---|
| $W_{qkv}^{(r)}$ storage + fused GEMM | `QKVParallelLinear` (inherits from `ColumnParallelLinear`) |
| `weight_loader(W_q, "q")`, `weight_loader(W_k, "k")`, `weight_loader(W_v, "v")` | `QKVParallelLinear.weight_loader` at offsets $0,\, H_q D / N,\, 2 H_q D / N$ |
| $\mathtt{qkv.split([\dots], dim=-1)}$ | `TPMHA.forward` — explicit sizes, not `chunk` |
| RoPE + SDPA per-rank heads | `TPMHA.forward` middle block |
| $\mathrm{attn}_r\, (W_o^{(r)})^{\top}$ + `all_reduce` | `RowParallelLinear` (already in ex01) |

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

## Aside — a code-organization pattern you'll meet in production

By ex03 you have four TP linear classes in this course:
`ColumnParallelLinear`, `RowParallelLinear`, `MergedColumnParallelLinear`,
`QKVParallelLinear` — and ex04 will add a fifth. Each hardcodes its
sharding axis: Column-family slices `dim 0` of `[out, in]`, Row slices
`dim 1`. Two dedicated implementations of the shard math live in two
different `weight_loader` bodies.

In production codebases where the class count is much higher (nanovllm's
[layers/linear.py](../../nanovllm-jun/nanovllm/layers/linear.py) has 5+
variants, Megatron's has 8+), that duplication becomes a maintenance
burden. The refactor everyone converges to is a single `LinearBase` that
takes a `tp_dim` attribute:

```python
class LinearBase(nn.Module):
    def __init__(self, ..., tp_dim: int | None = None):
        self.tp_dim = tp_dim  # 0 for column, 1 for row, None for replicated

    def weight_loader(self, param, loaded_weight):
        # One implementation that handles both dims via narrow(self.tp_dim, ...)
        shard_size = param.data.size(self.tp_dim)
        start = self.tp_rank * shard_size
        loaded = loaded_weight.narrow(self.tp_dim, start, shard_size)
        param.data.copy_(loaded)
```

Then `ColumnParallelLinear`, `RowParallelLinear`, and their variants
become thin subclasses that only set `tp_dim` at construction time
(`0`, `1`, or `None`).  This is what `nanovllm-jun/nanovllm/layers/linear.py`
does — see [`LinearBase`](../../nanovllm-jun/nanovllm/layers/linear.py) and
how each subclass just passes `tp_dim` upward.

**We keep the classes distinct in the bootcamp for pedagogical clarity**:
you can see "column shards dim 0, row shards dim 1" as a hardcoded fact
side-by-side, not as an attribute you have to trace. When you port back
to nanovllm-jun in the second week of your workshop project, you'll
refactor toward the `tp_dim` pattern — recognize it when you meet it,
but don't feel obligated to introduce it here.
