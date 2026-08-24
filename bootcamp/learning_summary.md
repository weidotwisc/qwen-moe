# Bootcamp learning summary

Personal notes on what I picked up in each exercise. Grows as I finish more.

## Progress

| Ex | Topic | Status | Tests | Notes |
|---|---|---|---|---|
| Ex00 | torch.dist primer (init/destroy + 5 wrappers) | ✅ done | 24/24 green | – |
| Ex01 | Linear TP (Column + Row) | ✅ done | 16/16 green | Refactored while doing Ex02 — see §5 of Ex02 for the fixes |
| Ex02 | SwiGLU MLP under TP (merged column + row) | ✅ done | 8/8 green | Forced the Ex01 refactor |
| Ex03 | MHA under TP (QKV column + O row) | ✅ done | 8/8 green | Introduced string `shard_id` + RoPE layout convention |
| Ex04 | GQA + KV replication | ✅ done | 8/8 green | **Nanovllm-jun bug fixed** — TP-8 on Qwen3-30B-A3B unblocked |
| Ex05a | MoE baseline (naive per-expert loop) | ✅ done | 2/2 green | Single-GPU. Consolidated `torch.where` / `index_add_` / broadcasting fluency |
| Ex05b | MoE permuted (grouped compute) | ✅ done | 2/2 green | Introduces the argsort/bincount/offsets vocabulary the rest of the MoE stack rides on |
| Ex06_ep | Expert Parallelism (replicated-input, canonical dispatch) | ⏸ reference-only | ref 8/8 green; **not implemented** | Framing conflates TP-scope with EP-scope — pattern belongs in Ex07 |
| Ex06_ep_lean | EP with single all_reduce (replicated-input) | ⏸ reference-only | ref 8/8 green; **not implemented** | Same TP-scope conflation as Ex06_ep — pattern belongs in Ex07 |
| Ex06_ep_pure | Pure EP with DP-partitioned inputs | ✅ done | 8/8 green | Dispatch structurally necessary — EP's natural form |
| Ex07 | TP + EP hybrid | ✅ done | 2/2 green (fp32 + bf16) | Case-3 canonical schedule — TP stripe + EP dispatch + TP all_gather |
| Ex08 | Weiz hybrid schedule (intra-lean + inter-dispatch) | ⏸ tabled | ref 2/2 + fused 2/2 green; benchmarks done | Correct but roughly tied w/ Ex07 under contiguous-TP topology — [[project_ex08_topology_finding]] |
| Ex09 | Fused MoE Triton kernel (grouped GEMM) | ⬜ pending | – | Ports Ex05b permuted MoE to Triton. Interview + paper artifact. |

**Bugs-caught count so far**: 26 distinct bugs across Ex01 + Ex02 + Ex03 + Ex04 + Ex05a + Ex05b + Ex06_ep_pure + Ex07 — see the "Traps I hit" sections in each entry for the failure mode and fix.

---

## Ex00 — `torch.distributed` primer

Seven tasks: NCCL init/teardown + five collective wrappers. Goal was to
build torch.dist syntax fluency and lock down the specs (pre/post
conditions) that the paper's proof story will treat as atomic events.

### 1. NCCL init/teardown

Every distributed script needs this at the top of each rank's worker:

```python
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
torch.cuda.set_device(rank)                       # MUST precede init
dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

**Non-obvious gotchas:**

- **`set_device` before `init_process_group`.** NCCL binds each rank to
  whichever device is current *at init time*. If you forget or reorder,
  two ranks can end up on the same GPU → NCCL will **silently deadlock or
  corrupt buffers on the first collective**, not error.
- **`MASTER_ADDR/PORT` are env-var-based rendezvous.** `init_process_group`
  with no explicit `init_method=` defaults to `env://`, which reads these
  variables. Without them: `ValueError: environment variable MASTER_ADDR
  expected, but not set`.
- **Port rotation.** Back-to-back tests hit TIME_WAIT on the same port;
  bump the port per run to avoid collisions.
- **`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`**: NCCL raises a Python exception
  on the offending rank instead of hanging when a collective fails
  mid-run. Was `NCCL_ASYNC_ERROR_HANDLING` before torch 2.2 (old name
  still works but emits a deprecation warning).

**What happens inside `init_process_group`:**

1. Rank 0 opens a TCP listener on `MASTER_PORT`.
2. Ranks 1..N-1 connect as clients.
3. TCPStore barrier: everyone waits for all `world_size` ranks. Any
   missing rank → everyone blocks here forever. This is the *single*
   place a distributed job can hang before the first collective.
4. NCCL sets up its device-to-device connections and returns.

Roughly `MPI_Init`, except MPI hides these steps behind the launcher
(`mpirun`) while torch.dist makes them explicit.

**`destroy_process_group`** should be idempotent-guarded (`if
dist.is_initialized(): dist.destroy_process_group()`) so a `finally:`
teardown doesn't mask an earlier exception with a second one.

### 2. The five collectives

| Wrapper | MPI equivalent | Notes |
|---|---|---|
| `all_reduce_sum(x)` | `MPI_Allreduce(SUM)` | In-place, returns None |
| `all_gather_into_tensor_wrapper(x)` | `MPI_Allgather` | Prefer over legacy `dist.all_gather` |
| `reduce_scatter_sum_tensor_wrapper(x)` | `MPI_Reduce_scatter_block(SUM)` | Inverse of allgather+sum |
| `all_to_all_equal(x)` | `MPI_Alltoall` | The "transpose" pattern |
| `all_to_all_variable(x, in_splits, out_splits)` | `MPI_Alltoallv` | **THE EP dispatch primitive** |

**Two important structural rules across the modern collectives:**

- **Destination buffer first.** torch's newer single-buffer collectives
  (`all_gather_into_tensor`, `reduce_scatter_tensor`, `all_to_all_single`)
  put the output tensor as the first positional arg — matches C's
  `memcpy(dst, src, n)` and CUDA's `cudaMemcpy`. **MPI is the outlier**
  (sender first). Mnemonic: torch = memcpy; MPI = mail delivery.
- **Caller allocates the output.** Unlike the legacy list-of-tensors
  `all_gather`, the modern variants demand a pre-allocated output
  buffer, same dtype + device as input, correct shape. For
  `all_to_all_variable` that shape is `(sum(output_split_sizes),
  *x.shape[1:])` — data-dependent leading dim.

### 3. `all_to_all_variable` — the EP dispatch primitive

The one that's actually subtle. Every rank has a 1-D-along-axis-0 input
tensor pre-sorted into `world_size` contiguous pieces by destination:

```
input_split_sizes[j] on rank i = rows this rank sends to rank j   (= MPI's sendcounts)
output_split_sizes[i] on rank i = rows this rank receives from i  (= MPI's recvcounts)
```

**Load-bearing invariant** (transposition):

```
input_split_sizes[j] on rank i  ==  output_split_sizes[i] on rank j
```

If violated even for one (i, j) pair → NCCL deadlocks or silently
produces garbage. It does NOT validate.

**Counts are measured in ROWS on axis 0**, not bytes or scalars. A
`[100, 2048]` tensor with `input_split_sizes=[20, 30, 50]` means "20
tokens (each of hidden=2048) to rank 0, 30 to rank 1, 50 to rank 2." No
element-count multiplication needed.

**How `output_split_sizes` is negotiated in real code** (nobody hand-writes
it):

1. Rank locally computes `input_split_sizes` (in EP: from `dest = expert_id
   // experts_per_rank` + `bincount`).
2. `all_gather` those vectors into a `[W, W]` matrix.
3. Rank `i` reads column `i` → its `output_split_sizes`.

nanovllm-jun's `_forward_expert_parallel` does exactly this — worth
re-reading before Ex06.

### 4. Process spawn patterns

Two dominant patterns, wildly different semantics:

| | `mp.spawn` (our test harness) | `torchrun` (Accelerate / FSDP / production) |
|---|---|---|
| Parent | The Python script itself | The `torchrun` CLI (a subprocess of the shell) |
| Spawn | Python `multiprocessing` — bootstrap re-imports | OS-level: N independent `python your_script.py` |
| Rank source | First positional arg (mp passes it) | `os.environ["LOCAL_RANK"]` / `RANK` / `WORLD_SIZE` |
| Rendezvous | You set `MASTER_ADDR/PORT` explicitly | torchrun sets env for children |
| Stderr per rank | Interleaved into parent | Each subprocess has its own stream |
| Failure handling | You catch; mp.spawn re-raises | Elastic mode: torchrun restarts failed workers |
| Use case | Test harnesses, notebooks, small experiments | Production training, distributed inference |

HF Accelerate is a thin wrapper over torchrun. FSDP itself doesn't launch
processes — you use torchrun (or Slurm's `srun`, or bare `mp.spawn`) and
then call `FSDP(model)` inside your script.

nanovllm-jun does something in-between: `multiprocessing.Process` +
`SharedMemory` for command passing (vLLM pattern). Long-lived workers,
dispatcher on rank 0.

### 5. Verification-friendly discipline

Habits I'm applying from Ex00 onward to make the paper's proof story
tractable:

- **Fixed collective schedule.** Every module issues the same sequence
  of collectives every call, no `if flag: coll_a else: coll_b`. Data-
  dependent branching around a collective is a deadlock hazard the
  paper's model would need to reason about; better to eliminate it.
- **Explicit `group=` args on every collective.** Even for the world
  group. Makes sub-group correctness syntactically obvious in Ex07.
- **Typed wrappers with docstring pre/post conditions.** Each wrapper
  becomes an atomic event in the paper's abstract model.
- **Progress-preserving loops.** Empty batches call `F.linear` with
  zero-row inputs rather than `continue` when a collective is anywhere
  in scope.
- **Deterministic reduction order** in the correctness path.

### 6. Python idiom notes worth internalizing

- **In-place mutation returns None**, per Guido / `list.sort()` /
  `random.shuffle()`. This is a mutation-visibility principle: returning
  `self` from a mutating method tricks the reader into thinking a new
  value was produced. `all_reduce_sum` returns `None`; `x` is mutated.
- **`torch.empty` beats `torch.zeros` for buffers that will be
  immediately overwritten by a collective.** Avoids a full-buffer write
  before the collective's own write. Not visible at small sizes; matters
  when hidden dims scale.
- **PEP 8 kwargs**: `op=dist.ReduceOp.SUM`, not `op = dist.ReduceOp.SUM`.
- **`torch.empty(x.shape[1:], ...)`** is safer than `torch.empty(*x.shape[1:],
  ...)` — the star-unpack silently breaks for 1-D input where
  `x.shape[1:]` is empty.

### 7. Collectives-as-functions: the mental model behind torch's naming

torch and MPI use different vocabularies for the same primitives, and
the difference is not cosmetic — it reflects two different mental
models:

| Vocabulary | Mental model |
|---|---|
| **MPI: `sendbuf` / `recvbuf` / `sendcounts` / `recvcounts`** | Collective = coordinated message-passing between peers |
| **torch: `input` / `output` / `input_split_sizes` / `output_split_sizes`** | Collective = pure function: `output = collective(input, ...)` |

Under torch's framing, a collective is **a function/channel** — every
rank calls it with its `input` tensor and receives its `output` tensor
back. What happens **inside** the function (send/recv, ring/tree
algorithms, RDMA path) is an implementation detail the caller doesn't
see. The naming is **operation-local**: `input`/`output` refer to
THIS call's flow, not to any pipeline of prior calls.

This has three concrete consequences I keep tripping over:

**(a) `input_split_sizes` and `output_split_sizes` are local to the
current call.** When implementing `combine = reverse(dispatch)`, my
instinct was to keep the same variable names for the same
"originally-sent" data and pass them through. Wrong.

For **dispatch**: `input=send_counts, output=recv_counts` — I'm
sending `send_counts[j]` to rank j, receiving `recv_counts[i]` from
rank i.

For **combine** (reverse): I now send back what I received. So
`input=recv_counts, output=send_counts`. **Swap the split lists.**
The names `input`/`output` don't inherit meaning from the prior call —
they always describe THIS call's send/receive counts.

Mnemonic: **each collective is standalone**. The reader who lands on
this line without seeing the previous call must be able to describe
what happens from `input_split_sizes` and `output_split_sizes` alone.

**(b) The function view enables autograd.** Because collectives look
like `output = f(input)`, torch can differentiate through them.
`all_reduce`'s backward is another `all_reduce` (identity on gradient);
`all_gather`'s backward is `reduce_scatter`. If we'd named things
`sendbuf`/`recvbuf` a la MPI, this would feel like a category error
("we're differentiating a message-passing op??"). Under the function
view it's just the chain rule for one more op.

**(c) The output is caller-allocated but conceptually the function's
result.** `dist.all_to_all_single(output, input, ...)` looks like it
takes two tensors, but semantically the `output` is a return value
(pre-allocated because torch's newer collective API demands it — no
implicit allocation). Think of it like `out=` in `torch.matmul(a, b,
out=y)` — the caller allocates the destination but conceptually the
op's producing it.

### The naming translation table

| MPI vocabulary | torch vocabulary | What it is |
|---|---|---|
| `sendbuf` | `input` tensor | The tensor this call sends out |
| `recvbuf` | `output` tensor | The tensor this call receives into |
| `sendcounts` | `input_split_sizes` | How many rows to each destination |
| `recvcounts` | `output_split_sizes` | How many rows from each source |
| `sendtype` / `recvtype` | (inferred from tensor.dtype) | — |
| `comm` | `group` | Which process group |

**Both are internally consistent under their own mental models.** MPI's
is mechanism-forward (bytes moving between peers); torch's is
interface-forward (function producing a result). For reading torch
code, commit fully to the function view — don't translate through
MPI's send/recv paradigm en route.

### Reference material worth revisiting

- [bootcamp/ex00_dist_primer/README.md](ex00_dist_primer/README.md) — MPI ↔ torch.dist mapping table.
- [bootcamp/ex00_dist_primer/solution.py](ex00_dist_primer/solution.py) — the specs I wrote.
- nanovllm-jun's `_forward_expert_parallel` — real-world EP dispatch using these primitives.

---

## Ex01 — Linear Tensor Parallelism

**Status: done, 16/16 tests green.** Refactored during Ex02 to
pre-allocate `self.weight` in `__init__` (see §5 below for the log of
the original deferred plan, and Ex02 §5 for the actual resolution).

Two classes: `ColumnParallelLinear` (shards output dim) and
`RowParallelLinear` (shards input dim). One collective total per
`Column → Row` pair (the row's all-reduce). The math is easy; the
conventions around it are where all the friction lives.

### 1. Sharding math

Full weight lives at shape `[out_features, in_features]` in PyTorch's
convention. Each rank holds a slice:

| Class | Slice on | Per-rank shape | Collective in forward |
|---|---|---|---|
| `ColumnParallelLinear` | dim 0 (out) | `[out/N, in]` | none — output stays sharded |
| `RowParallelLinear`    | dim 1 (in)  | `[out, in/N]` | one `all_reduce(SUM)` on the partial output |

Composed: `Column → Row` gives one all-reduce per pair, which is why
MLPs are shaped as `col → row` (gate_up → down), not the reverse. The
row's all-reduce is the *only* cross-rank comm in a whole MLP.

Under Axler's `y = Wx` mental model: **ColumnParallel = shard rows of W
= each rank computes some rows of `y`**; **RowParallel = shard columns
of W = each rank contributes a partial sum**, all-reduce sums the
partials.

### 2. Convention landscape (this is the pedagogical content)

The exercise is *simple*; the conventions are what took time to sort.

- **PyTorch stores `W` in `[out, in]`.** This matches classical
  linear-algebra `y = Wx` where inputs are column vectors. `.weight.shape`
  reports this honestly.
- **`F.linear(x, W)` computes `x @ W.T`** (batched row-vector inputs
  need a `.T` on `W`). The transpose is a zero-copy view; cuBLAS
  applies it via the `transb=True` flag with no data movement.
- **Constructor and shape disagree on ordering.** `nn.Linear(in, out)`
  is data-flow order (input→output). `.weight.shape == (out, in)` is
  matrix-notation order (rows-first). Both mathematically legitimate,
  jarring together.
- **Megatron's `A = W.T` framing** in the paper vs W-storage in the code
  is a *known* convention gap, not a bug. Megatron's own code has this
  comment: *"torch.nn.functional.linear performs XA^T + b and as a
  result we allocate the transpose."*  ([Megatron layers.py L969](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L969)).
- **Other frameworks disagree.** JAX/Flax's `Dense`, Keras, and CMU
  10-714's needle store weight in `[in, out]` (A-shape) — cleaner spec
  ↔ code correspondence. PyTorch is the awkward outlier for legacy
  BLAS/Torch7 reasons. The bootcamp sticks with PyTorch's convention
  for ecosystem alignment.
- **My chosen mental model**: Axler-camp. Trust `.shape` = W-shape,
  read Megatron's "columns of A" as "rows of W" on the fly.
- **Naming translation**: `ColumnParallel` = shard the *out* dim
  (whether you call that "columns of A" or "rows of W"). `RowParallel`
  = shard the *in* dim. When in doubt, name axes by their semantic role
  (out vs in), not by "row/column."

### 3. Traps I hit

- **Operator precedence in shard-index math.** `tp_rank // tp_size * N`
  ≠ `tp_rank * N // tp_size`. Python evaluates `//` and `*` left-to-right
  (same precedence). If `tp_rank < tp_size` — always true — then
  `tp_rank // tp_size = 0` and the whole expression collapses to 0 or
  `N`, silently producing empty or full slices. My first ColumnParallel
  passed only the `tp_size=1` cases (2/8 tests). The fix is
  `tp_rank * (N // tp_size)` — parenthesize what you mean.
- **`self.weight` should be an `nn.Parameter` allocated in `__init__`
  with `torch.empty(shape, ...)`, then filled by `weight_loader.data.copy_(shard)`.**
  I initially created `nn.Parameter(shard)` *inside* `weight_loader`
  (for Column) or assigned a raw tensor (for Row). Both pass Ex01's
  tests because `.to()` is called before `weight_loader` and full_weight
  is already on the target device/dtype. But this pattern:
  1. Doesn't participate in `.to(device, dtype)` after construction.
  2. Isn't visible to `nn.Module.parameters()` / optimizer / hooks /
     `torch.compile` guards.
  3. **Will break in Ex02**, where `MergedColumnParallelLinear.weight_loader`
     is called twice at different offsets — the pattern needs the same
     pre-allocated buffer both times.
- **Inconsistent attr naming.** I used `self.params` in Column but
  `self.weight` in Row. Both work for Ex01, but Ex02's subclass expects
  `self.weight` on Column. Pick one name (the standard is `self.weight`).

### 4. Verification framing

- **`.shape` is the abstraction boundary.** Whichever convention you
  pick for weight storage, `.shape` is the honest source of truth. The
  paper's `A` never exists as a real `Parameter`; it's a narrative device
  for the abstract spec. The transpose lives entirely in `weight_loader`
  (as a `.T` at load time if you chose A-space storage) or entirely in
  `forward` (as a `.T` view in `F.linear`, if you chose W-space).
- **Sharding rule as a spec**: "at end of `weight_loader`, rank r's
  `self.weight.data == full_weight.narrow(shard_dim, r * shard_size,
  shard_size)`."  Two lines, straightforward Verus predicate.
- **Forward invariant**: after `RowParallelLinear.forward` returns,
  `y` is *replicated* across the TP group. Before the `all_reduce`, it
  was *partial-sum-sharded*. That single all-reduce is the state
  transition.
- **The 4-phase `nn.Module` lifecycle is what the Verus spec quantifies
  over.** Every parallel module has the same lifetime:

  | Phase | Trigger | Post-condition |
  |---|---|---|
  | 1. Construction | `Module(...)` | Parameter allocated with `torch.empty(shard_shape)`; on default device/dtype; **uninitialized bytes** |
  | 2. Placement | `.to(device, dtype)` | Parameter re-materialized on target device/dtype; still uninitialized |
  | 3. Loading | *external* `weight_loader(full_weight [, shard_id])` | `param.data == full_weight.narrow(shard_dim, r * shard_size, shard_size)` |
  | 4. Inference | `layer(x)` | Reads Parameter; returns `y` (sharded or replicated per class); Parameter unchanged. Repeatable. |

  Two structural facts encoded here:
    - **Phase 3 is externally driven.** The module never calls its own
      `weight_loader`; the test / checkpoint walker does. The abstract
      state machine has one transition per external call, not per
      internal method invocation.
    - **Between Phases 2 and 3, the module is in an
      "allocated-but-uninitialized" state**. Forward called here would
      produce garbage. The spec's precondition on Phase 4 is
      *"Phase 3 has completed at least once for every Parameter."*

  The paper's proof for TP correctness can be structured as:
  *(a) Phase 3's post-condition ⇒ (b) Phase 4's forward preserves the
  layer's semantic contract (`y ≈ x W^\top` up to reduction order).*
  Two Hoare triples, one per phase. Clean composition point.

### 5. Deferred: the pattern overhaul (revisit after Ex02)

**Update: resolved after Ex02** — Ex02's merged-variant inheritance
forced the fix on the same evening. See Ex02 §5 below for the actual
resolution. The section below is preserved as historical record of
what was originally planned.

My Ex01 originally:
- creates `nn.Parameter` inside `weight_loader` (Column) or assigns raw
  tensor (Row);
- uses `self.params` on Column and `self.weight` on Row;
- lacks the divisibility assert on `RowParallelLinear.__init__`.

All three will be fixed **once Ex02 forces the issue** — the merged
variant needs a pre-allocated buffer, uniform `self.weight` name, and
the invariants tightened. Then come back here and:

1. Move `self.weight = nn.Parameter(torch.empty(shard_shape))` into
   both classes' `__init__`.
2. Change both `weight_loader` bodies to `self.weight.data.copy_(shard)`.
3. Rename `self.params → self.weight` in Column.
4. Add `assert in_features % tp_size == 0` in Row's `__init__`.
5. Re-run `pytest bootcamp/tests/test_ex01_linear_tp.py` — should still be 8/8.

This "let the next exercise force the earlier fix" pattern is
deliberate — I learn the invariant by feeling the failure, not by being
told. Just don't forget to circle back before the paper's proof
artifact is finalized. The Ex01 code in the paper's artifact should
have all three fixes applied.

### Reference material worth revisiting

- [Megatron-LM layers.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py) — the self-aware `# we allocate the transpose` comment.
- [bootcamp/ex01_linear_tp/README.md](ex01_linear_tp/README.md) — the code-organization aside about `LinearBase` + `tp_dim` that production codebases converge to.
- [nanovllm-jun/nanovllm/layers/linear.py](../nanovllm-jun/nanovllm/layers/linear.py) — the reference for what my code will look like after the port back.

---

## Ex02 — SwiGLU MLP under TP

**Status: done, 8/8 tests green. Also forced the Ex01 refactor** —
Ex01 §5's deferred fixes were applied while working through this
exercise; details in §5 of this section.

First real composition: chain ex01's `ColumnParallelLinear` +
`RowParallelLinear` into a full SwiGLU MLP, with a **merged** variant
(`MergedColumnParallelLinear`) that fuses gate + up into a single
weight buffer.

### 1. Sharding math

Weights (unsharded):
$W_g, W_u \in \mathbb{R}^{I \times H}, \; W_d \in \mathbb{R}^{H \times I}$.

**Merged column shard**: vertically stack gate + up, then row-shard on
`tp_size`:

$$
W_{gu} = \begin{bmatrix} W_g \\ W_u \end{bmatrix} \in \mathbb{R}^{2I \times H},
\qquad
W_{gu}^{(r)} \in \mathbb{R}^{(2I/N) \times H}.
$$

Per-rank forward:

$$
\begin{aligned}
[G_r \mid U_r] &= X\,(W_{gu}^{(r)})^{\top}     & \text{shape } [B, 2I/N] \\
Z_r &= \mathrm{SiLU}(G_r) \odot U_r             & \text{shape } [B, I/N], \text{elementwise, no comm} \\
y &= Z_r\,(W_d^{(r)})^{\top} + \text{allreduce} & \text{shape } [B, H], \text{replicated}
\end{aligned}
$$

**Total comm: one all-reduce per forward** (inside RowParallel's
down_proj). This is the canonical "column → row" pattern's collective
budget.

### 2. The design pattern: N-projection merged columns

`MergedColumnParallelLinear(in_features, output_sizes: list[int], ...)`
generalizes to **N projections sharing one input `x`**. Three concrete
cases in the bootcamp:

| Case | N | `output_sizes` | Where |
|---|---|---|---|
| SwiGLU MLP | 2 | `[I, I]` | Ex02 (now) |
| MHA QKV | 3 | `[n_heads·D, n_heads·D, n_heads·D]` | Ex03 |
| GQA QKV | 3 | `[n_heads·D, n_kv·D, n_kv·D]` (unequal) | Ex04 |

Merging is legal iff the projections share their input. **No symmetric
`MergedRowParallel`** exists because row-parallel's structure
(partial-sum → all-reduce) has no N-way fusion opportunity — you'd need
N row projections summing into one target, and transformers don't have
that shape.

**Historical**: pre-gated MLPs (GeLU-style, Megatron 2019) didn't need
merged column parallel at all. vLLM (SOSP 2023) crystallized the class
hierarchy for gated MLPs + QKV. nanovllm inherited it verbatim. For the
Verus spec, one theorem parameterized by `output_sizes` covers SwiGLU
and QKV under the same framework.

### 3. Traps I hit

- **Operator precedence carried over from Ex01.** My first offset math
  *added the source-rank offset to the target offset*, placing each
  shard in the wrong location. Fix: the target and source offsets are
  independent — source is `full_weight.chunk(N, 0)[tp_rank]`, target is
  `sum(output_sizes[:shard_id]) // tp_size`, no addition between them.
- **Indexing an empty list in a build loop.** Wrote
  `self.slices = []` then `self.slices[shard_id].append(...)` — the
  outer indexing errors immediately because the list is empty. Meant
  `self.slices.append((offset, length))`. Basic mistake, easy fix once
  the traceback fingers the line.
- **`len` shadowing the built-in.** Python's LEGB scoping rule: *any*
  assignment to a name in the function body makes that name local for
  the entire function, from line 1 to line last. Naming a local
  variable `len` triggered `UnboundLocalError` on an earlier line where
  I called `len(...)` as the built-in. Fix: rename to `length`. Enabled
  `ruff` rule `A00X` (flake8-builtins) to catch this class of bug at
  write time going forward.

### 4. Verification framing

- **The 4-phase lifecycle carries over.** Phase 3 (weight_loader) is
  invoked **N times per Merged instance**, once per `shard_id`. Each
  call establishes a **disjoint sub-invariant** over its slice of the
  merged buffer. Post-condition after all N calls:

  $$
  \forall\, k \in [0, N):\;\; \text{param}[o_k : o_k + s_k,\, :] \;=\; \text{full}_k[r \cdot s_k : (r{+}1) \cdot s_k,\, :]
  $$

  where $o_k = \sum_{i<k} s_i / N$ (target offset) and $s_k = \text{output\_sizes}[k] / N$
  (per-rank shard size). Cleanly composable across projections.

- **Weight_loader calls commute.** Filling gate then up produces the
  same buffer state as up then gate — disjoint memory regions, no
  interference. For the Verus spec: the N Phase-3 calls can be modeled
  as an unordered set of transitions, not a strict sequence. That
  simplifies the proof — one less axiom about call ordering.

- **Composed-layer invariant across the SiLU⊙ boundary.** At the exit
  of `MergedColumnParallelLinear`, the intermediate tensor is
  *column-sharded on the last axis*. `RowParallelLinear`'s
  precondition (input sharded on in-dim) is exactly this
  post-condition, so composition is **immediate** — one Hoare triple
  chains straight into the next, no intermediate collective needed.
  This is what makes the whole MLP pay only one all-reduce.

### 5. Ex01 pattern overhaul — resolved

The three items I deferred in Ex01 §5 got fixed in the pass that
unblocked Ex02:

1. `self.weight = nn.Parameter(torch.empty(...))` allocated in
   `__init__` for both `ColumnParallelLinear` and `RowParallelLinear`.
2. Both `weight_loader` methods now use `self.weight.data.copy_(shard)`.
3. Renamed `self.params → self.weight` in Column for symmetry with Row.
4. Added `assert in_features % tp_size == 0` in Row's `__init__`.

All 16 Ex01 tests still pass after the refactor. Merged inheritance now
works cleanly — Ex02's `self.weight.narrow(...)` refers to a
pre-allocated buffer allocated by `super().__init__()`.

**Ex02 test result: 8/8 green.** Confirms both the merged pattern and
the Ex01 refactor work end-to-end.

### 6. Terminology observations worth noting for the paper

- **`shard_id` overloads "shard".** In TP it usually means "the 1/N
  slice held by rank r." In `MergedColumnParallelLinear.weight_loader`,
  it means "which of the N merged projections is this call for."
  Different concept, same word. Cleaner Verus terminology:
  `role_id` / `projection_index` for the merged case; keep `tp_rank`
  for the mechanical TP shard.
- **The naming is retroactively locked in** because vLLM shipped first
  and every downstream project inherits its API. Paper can safely
  rename in the abstract Verus model with a one-line refinement lemma;
  the concrete Python code stays as-is for ecosystem compatibility.

### Reference material worth revisiting

- vLLM's [`MergedColumnParallelLinear` in linear.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py) — the API origin.
- [Shazeer 2020, "GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) — the paper that motivated widespread gated MLPs, and thus the need for MergedColumnParallelLinear in the first place.
- [nanovllm-jun/nanovllm/layers/linear.py](../nanovllm-jun/nanovllm/layers/linear.py) — production shape my bootcamp code targets for the port back.

---

## Ex03 — Multi-Head Attention under TP

**Status: done, 8/8 tests green.** Introduces `QKVParallelLinear`
(N=3 merged column with equal-size projections for MHA) and `TPMHA`
composing QKV → RoPE → SDPA → RowParallel O.

### 1. Sharding math

Under TP-$N$ with $H$ heads and head_dim $D$:

$$
W_q, W_k, W_v \in \mathbb{R}^{H D \times \text{hidden}}, \qquad
W_o \in \mathbb{R}^{\text{hidden} \times H D}
$$

Merged QKV, per-rank shape $[3 (H/N) D, \text{hidden}]$:

$$
W_{qkv}^{(r)} = \begin{bmatrix} W_q^{(r)} \\ W_k^{(r)} \\ W_v^{(r)} \end{bmatrix} \in \mathbb{R}^{3 (H/N) D \times \text{hidden}}
$$

Per-rank forward (one collective total, from Row's O all-reduce):

$$
\begin{aligned}
[Q_r \mid K_r \mid V_r] &= X (W_{qkv}^{(r)})^\top && \text{shape } [B, T, 3 (H/N) D] \\
Q_r, K_r &\gets \mathrm{RoPE}(Q_r, K_r) && \text{applied in } [B, T, H/N, D] \text{ layout} \\
\mathrm{attn}_r &= \mathrm{SDPA}(Q_r, K_r, V_r) && \text{on } [B, H/N, T, D] \text{ layout after transpose} \\
Y &= \sum_r \mathrm{attn}_r (W_o^{(r)})^\top && \text{via one all-reduce inside Row}
\end{aligned}
$$

Same "column → row + one all-reduce" pattern as MLP (Ex02). Confirms
the block-level invariant: **attention and MLP have identical TP shape**,
their only difference is the per-shard operation between the two
projections (SiLU⊙ for MLP, SDPA for attention).

### 2. Design patterns worth noting

- **QKV as N=3 merged column** with equal sizes for MHA (`num_heads == num_kv_heads`).
  Same abstract pattern as Ex02's merged MLP, just N=3 instead of N=2.
  `output_sizes = [n_heads·D, n_heads·D, n_heads·D]`.
- **String `shard_id`** (`"q"`, `"k"`, `"v"`) — matches HF safetensors keys.
  Ex02's SwiGLU used int (`0`=gate, `1`=up) for the same abstract role.
  Both work; the string version is more self-documenting for QKV.
- **RoPE layout convention: `[B, T, H, D]`** (not `[B, H, T, D]`). We
  apply RoPE **before** the SDPA transpose, matching the older
  vLLM/LLaMA-1 style. Modern HF `modeling_qwen3_moe.py` transposes
  first — same math, opposite convention. Bootcamp matches vLLM for
  port-compatibility with nanovllm-jun.
- **SDPA needs `[B, H, T, D]`** — hard-required by every attention
  kernel (F.scaled_dot_product_attention, flash-attn, cuDNN, Xformers).
  Both convention styles converge here via a mandatory transpose. The
  only question is whether RoPE happens before or after that transpose.
- **`.reshape` after transpose**, not `.view`. The transposed
  `[B, T, H, D]` tensor may not be contiguous; `.reshape` handles the
  copy-if-needed transparently, `.view` would error.

### 3. Traps I hit

- **`torch.split(qkv, [q_size, kv_size, kv_size], dim=0)`** — splitting
  the batch dim instead of the feature dim. Would produce shape errors
  or silently wrong slices depending on values. Fix: `dim=-1`. Classic
  "which axis am I actually splitting" mistake, easy to catch by
  reading the shape at the call site.
- **Missing `return o` at end of `forward`**. Function fell off the
  end, returned `None`, test failed with a shape-of-None error. Trivial
  but real — the `TODO(you)` comment listed 9 steps and I stopped at
  step 9's action without adding the final return statement.

### 4. Verification framing

- **Block-level invariant**: `TPMHA.forward(x)` obeys **replicated in,
  replicated out** — same invariant as MLP. Interior sharding on heads
  is entirely encapsulated within the block.
- **Composition with MLP is trivial**: attention exits with $Y$
  replicated across TP group → LayerNorm operates elementwise on
  replicated $Y$ → residual add on replicated tensors → MLP receives
  replicated input → MLP exits with replicated output. One boolean
  invariant carried through the entire residual stream.
- **RoPE's spec is layout-invariant.** The abstract Verus predicate:
  "for each (batch, seq_pos, head, dim_pair), apply the position-
  dependent 2D rotation." Whether the Python code does this in BTHD
  or BHTD is an implementation detail below the spec's abstraction
  level — same theorem covers both conventions.
- **SDPA as an axiom.** Since `F.scaled_dot_product_attention` is a
  differentiable primitive with a full backward, the Verus spec treats
  it as an opaque operation with the well-known softmax-attention
  post-condition. No need to prove SDPA's internals; the theorem
  quantifies over "for a spec-conforming SDPA, TP-parallel attention
  produces the correct output." Same abstraction that flash-attn / cuDNN
  attention / Ring Attention all satisfy — proof transfers freely.

### 5. Terminology observations

- **Renamed `head_size` → `head_dim`** in `QKVParallelLinear`. vLLM /
  Megatron use `head_size`; HF and Qwen3 config use `head_dim`. Kept
  a one-line comment noting the alias, so a reader (or the intern
  porting to Verus) sees the correspondence explicitly.
- **`num_kv_heads` is `num_k_heads == num_v_heads`.** K and V heads
  always come in matched pairs — a KV head "owns" both a K projection
  and a V projection. Naming reflects this rather than pretending they
  could differ.

### Reference material worth revisiting

- HF's [`modeling_qwen3_moe.py::Qwen3MoeAttention`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) — modern transpose-first attention pattern (contrast to bootcamp's convention).
- vLLM's [`QKVParallelLinear` in linear.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py) — the source of the industry-standard API shape we're matching.
- [Meta's LLaMA-1 reference code](https://github.com/meta-llama/llama) — where the "RoPE before transpose" convention originated.

---

## Ex04 — GQA + KV-head replication under TP

**Status: done, 8/8 tests green. Real project milestone** — this
implementation fixes nanovllm-jun's `assert num_kv_heads % tp_size == 0`
failure at TP-8 on Qwen3-30B-A3B (num_kv_heads=4). The code here ports
back directly.

### 1. Sharding math

Given $H_q$ Q heads and $H_{kv}$ KV heads with $H_{kv} \le H_q$ and
$H_q \bmod H_{kv} = 0$. Under TP-$N$:

$$
\begin{aligned}
n_q^{(r)}   &= H_q / N   &&\text{(Q heads per rank)} \\
n_{kv}^{(r)} &= \max\!\left(1,\; H_{kv} / N\right)   &&\text{(KV heads per rank)} \\
r_{kv}      &= \max\!\left(1,\; N / H_{kv}\right)    &&\text{(replicas of each KV head)}
\end{aligned}
$$

Per-rank storage: $(n_q^{(r)} + 2 n_{kv}^{(r)}) D$ rows in the merged
QKV weight. Total across ranks: $N \cdot (n_q^{(r)} + 2 n_{kv}^{(r)}) D
= (H_q + 2 H_{kv} r_{kv}) D$.

Under replication, the total is **larger than the raw un-sharded weight
$(H_q + 2 H_{kv}) D$** — because KV is redundantly stored on $r_{kv}$
ranks. Memory cost, correctness gain (every rank can attend locally).

### 2. Design patterns worth noting

- **The `max(1, ...)` clause is the whole KV-replication fix.**
  `n_{kv}^{(r)} = max(1, H_{kv} / N)` clamps to 1 head per rank when
  $N > H_{kv}$; the `r_{kv} = max(1, N / H_{kv})` clause tells how many
  ranks share each KV head. One conditional-free formula covers both
  regimes.
- **Unified chunk math**: the `weight_loader` uses
  `full_kv.chunk(tp_size // kv_replicas, dim=0)[tp_rank // kv_replicas]`.
  Under normal sharding (`kv_replicas=1`), this is
  `chunk(tp_size)[tp_rank]` — identical to ex03. Under replication
  (`kv_replicas>1`), it becomes `chunk(H_{kv})[tp_rank // r_{kv}]`,
  which gives multiple ranks the same slice. One code path for both
  regimes.
- **KV replication = implicit DP sub-group inside TP.** Under training
  (not our scope), the replica group would need `all_reduce(SUM)` on
  the KV weight gradients to keep replicas byte-identical. That's a
  DDP-shaped operation nested inside a TP framework — a two-axis
  parallelism revealed by the design. For inference (Ex04), invisible
  — no backward, no gradient sync.
- **`repeat_interleave` after RoPE, before SDPA** (or equivalent
  broadcasting): required to bring K/V head count up to Q's before
  SDPA. Can happen either in BTHD layout on `dim=2` (reference style)
  or in BHTD layout on `dim=1` (my solution). Both correct;
  materialize the same duplicated tensor. Modern PyTorch 2.6+
  auto-detects GQA in SDPA and would skip the materialization —
  future perf work.

### 3. Traps I hit (5 bugs, in the order I hit them)

- **`out_features` conceptual mismatch: per-rank vs global.**
  My first `super().__init__(hidden, self.q_shard + self.k_shard + self.v_shard, tp_size, tp_rank, group)`
  passed the per-rank size where the parent expected the total. Parent
  then allocated `self.weight` with shape `(per_rank / tp_size,
  in_features)` — an under-sized buffer. Fix: multiply by `tp_size`
  to get the global total: `out_features = tp_size * (q_shard + k_shard + v_shard)`.
- **`RowParallelLinear.in_features` same conceptual mistake.** Passed
  `n_heads_per_rank * head_dim` (per-rank input width) where the
  parent expects the total across ranks. Both parent classes take
  **global** dims and divide internally.
- **Missing `super().__init__()` call.** I put `self.hidden = ...`
  and `self.weight = nn.Parameter(...)` in `QKVParallelLinearGQA.__init__`
  without ever calling the parent's `__init__`. Result:
  `AttributeError: cannot assign parameters before Module.__init__() call`
  — the `nn.Module` framework needs its `_parameters` dict initialized
  before any Parameter attribute can be assigned. C++-style thinking
  (base class constructor is auto-called) doesn't apply — in Python,
  `super().__init__()` is a manual invocation.
- **Missing `repeat_interleave` on K, V** in TPGQA.forward. Under
  GQA with `num_kv_heads < num_heads` per rank, K and V have fewer
  heads than Q. SDPA (in most PyTorch versions) requires matched Q/K/V
  head counts. Fix: `k.repeat_interleave(n_rep, dim=1)` (BHTD) or
  `k.repeat_interleave(n_rep, dim=2)` (BTHD) to broadcast KV heads up
  to Q's head count. Either placement works; both materialize the
  duplicated tensor.
- **`group=group` missing on `o_proj`.** Only `qkv_proj` got the
  group; `o_proj` defaulted to the world group. Currently invisible
  because test's world group == TP group, but silently wrong under
  ex07's TP+EP hybrid where TP is a sub-group of world.

### 4. Verification framing

- **Ex04 is a strict superset of Ex03's spec.** Under the special case
  $H_{kv} = H_q$ and $N \le H_{kv}$, Ex04's math degenerates to Ex03's
  MHA. The `max(1, ...)` clause is inert in that regime. So the
  Verus theorem for Ex04 subsumes Ex03's — one proof covers both.
- **The KV-replication invariant is a bijection property**: at any
  point in the model's lifecycle, KV heads on replica ranks have
  **identical byte values**. Post-Phase 3 (weight_loader), this is
  established by the loader giving identical slices to co-replicas.
  Post-forward, unchanged (weights are read-only). Post-optimizer-step
  (training, not our scope), preserved only if the replica-group
  gradient all-reduce runs before the step.
- **Attention math per rank is *identical* to a single-GPU GQA**
  restricted to this rank's Q heads. No cross-rank dependency during
  the attention math itself (the O all-reduce is the only cross-rank
  event, and it's post-attention). This is the "block-level invariant"
  from ex01-03 preserved through GQA.

### 5. Prior-exercise cross-cutting lesson: global vs per-rank

Two bugs in Ex04 (out_features + in_features) had the same root:
**parent classes take *global* dims and divide internally by tp_size**.
The pattern:

- Constructor: `Parent(in_features=<global>, out_features=<global>, tp_size, tp_rank)`.
- Internal: allocates `weight` of shape `<global> // tp_size` per rank.
- Caller: passes the semantic model dims (`hidden`, `num_heads * head_dim`),
  not the sharded values.

This is the correct convention (matches `nn.Linear` and industry TP
practice) but requires discipline. Whenever a subclass computes a total
"across all ranks," check twice that you're multiplying rather than
dividing.

### 6. Terminology observations

- **`kv_replicas` (my naming) vs `num_kv_replicas`** — the reference
  uses the longer name for clarity. Either is fine; consistent naming
  within your codebase is what matters.
- **`num_q_heads_per_rank` vs `num_heads_per_rank`** — I disambiguated
  by including `q` in the name because Ex04 has two "per-rank" head
  counts (Q and KV). Reference used just `num_heads_per_rank` for Q.
  The disambiguated form is more explicit; matter of style.

### Reference material worth revisiting

- [nanovllm-jun `Qwen3Attention.__init__`](../nanovllm-jun/nanovllm/models/qwen3.py) — the exact `assert num_kv_heads % tp_size == 0` line this exercise's KV replication fixes.
- HF's [`Qwen3MoeAttention`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) — production GQA implementation without KV replication (they don't need it because HF forward is not TP-sharded).
- PyTorch 2.6+ [`F.scaled_dot_product_attention` GQA auto-detection](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — the perf optimization that skips `repeat_interleave` when Q/K head counts differ.

---

## Cross-cutting: TP vs FSDP / Zero-Inference for LLM serving

This came up while thinking about DeepSpeed-MII. Worth extracting from
the exercise-by-exercise notes because it's a general framing that
guides the whole paper's positioning.

### Core insight

**TP transfers activations. FSDP / Zero-Inference transfers weights.**
For a model that fits under TP-$N$, TP is dramatically more
bandwidth-efficient because activations are much smaller than weights
at typical inference workloads.

### The asymmetry, quantified

For Qwen3-30B-A3B under 8-way parallelism:

|  | TP-8: activation comm | ZI-8: weight comm |
|---|---|---|
| Per forward at batch=1, seq=1 (decode) | ~400 KB | ~42 GB |
| Per forward at batch=64, seq=1024 (prefill) | ~25 GB | ~42 GB |
| Per forward at batch=256, seq=1024 | ~100 GB | ~42 GB |

Weight comm is **fixed** — it doesn't depend on batch or sequence
length. Activation comm scales with `batch × seq`. At decode-scale,
TP moves ~100,000× less data than ZI. At heavy prefill, they get
comparable. At extreme long-context / high-batch, ZI wins the comm
race.

### The crossover threshold

The point where TP activation comm equals ZI weight comm:

$$
B \cdot T \;\approx\; \frac{\tfrac{N-1}{N} \cdot \text{total\_params}}{2 \cdot \text{hidden} \cdot \text{n\_layers}}
$$

For Qwen3-30B-A3B (30B params, hidden=2048, 48 layers, TP-8, bf16):

$$
B \cdot T \;\approx\; 130{,}000 \text{ tokens/forward.}
$$

Below this threshold, TP is bandwidth-cheaper. Above, ZI is. **Typical
inference is well below** (batch × seq usually in the thousands, not
hundreds of thousands), so TP dominates practically all realistic
inference workloads.

### The general principle: parallelism trades one comm axis for another

| Scheme | What flows across ranks | Cost scales with |
|---|---|---|
| **TP** | activations | $B \cdot T \cdot \text{hidden}$ |
| **FSDP / ZI** | weights | fixed = model size |
| **PP** | activations at layer boundaries | $B \cdot T \cdot \text{hidden} \cdot N$ |
| **EP** (MoE) | tokens (post-routing) | $B \cdot \text{top-}K \cdot \text{hidden}$ |
| **DP** (training only) | gradients | fixed = model size |

Each scheme is optimal in a different regime. **TP wins when
activations ≪ weights**, which is the LLM-inference regime because of
architectural facts: `hidden ≪ total_params / num_layers`, and inference
batches are small.

### Why single-GPU is a stiff baseline (surprisingly)

For a model that fits on one GPU, single-GPU inference is very
efficient: 100% HBM bandwidth utilized, zero cross-GPU comm. Any
distributed scheme has to overcome its own comm overhead to catch up:

- **TP-N**: still ~90-95% efficient because activation comm is tiny.
- **ZI-N**: efficiency drops significantly at low batch because
  weight comm is 100,000× activation comm. For a model that fits under
  TP, ZI-N is often **slower than single-GPU** at low batch.

So the "distribute to speed up" reflex isn't automatic — you have to
check that your comm cost is less than the compute win.

### When ZI / FSDP wins (only when the model doesn't fit under TP)

- **175B / 405B / 671B models** on 8×80GB clusters where TP-8 alone
  can't fit them. ZI's memory-per-rank scales as `total_params / N`,
  so at N=8 you cut memory 8× further via ZI-3 layout. Models TP-8
  can't hold, ZI-8 can.
- **Very long-context prefill** (>>100K tokens per pass) where
  activation memory itself is the bottleneck. ZI can offload
  activations too via `activation_checkpointing`.
- **Fine-tuning + inference in the same framework**: switch cost of
  leaving DeepSpeed / FSDP for a TP-only inference server is high.

For everything else — TP wins.

### Consequence for this paper

The paper's target (nanovllm with TP+EP for Qwen3-30B-A3B) is exactly
the workload where TP dominates: mid-sized model that fits under TP-8,
inference-only, moderate batches. **Choice of TP+EP is not arbitrary
— it's dictated by the workload's bandwidth economics.** The paper's
methodology section can state this in one paragraph: verify the
parallelism scheme that industry already uses for this class of
workload.

### Reference

- ZeRO paper (Rajbhandari et al., SC 2020) — introduces the sharding
  levels ZeRO-1/2/3.
- DeepSpeed-Inference blog (2022) — TP with fused kernels.
- DeepSpeed Zero-Inference (2022) — extends Zero-3 to inference with
  CPU/NVMe offload.
- FSDP paper (Zhao et al., VLDB 2023) — PyTorch's Zero-3 equivalent.

---

## Ex05a — Naive Sparse MoE (single-GPU, per-expert loop)

**Status: done, 2/2 tests green.** Single-GPU baseline. No TP, no EP —
this exercise is about the sparse routing math + per-expert loop
pattern that HF's `Qwen3MoeSparseMoeBlock` uses in production. Ex05b
will re-implement the same math in the permuted layout; Ex06 will
bolt all-to-all onto that permuted variant.

### 1. Sparse MoE math

Given tokens $X \in \mathbb{R}^{N \times H}$, router $W_R \in \mathbb{R}^{H \times E}$,
and $E$ SwiGLU MLP experts $\{f_e\}$:

$$
\begin{aligned}
G &= \mathrm{softmax}(X W_R) \in \mathbb{R}^{N \times E} & \text{routing probs (fp32 in-kernel)} \\
(w_{ij}, e_{ij})_{j=1..k} &= \mathrm{topk}(G_i, k) & \text{per-token top-k experts} \\
w_{ij} &\gets w_{ij} \Big/ \textstyle\sum_{j'} w_{ij'} & \text{if norm\_topk\_prob} \\
Y_i &= \textstyle\sum_{j=1}^{k} w_{ij}\, f_{e_{ij}}(X_i) & \text{weighted combine}
\end{aligned}
$$

Per-token top-k = "each token uses only k of the E experts." Weighted
combine = "sum the k expert outputs with softmax weights." That's the
whole MoE forward.

### 2. The `softmax(topk(logits))` equivalence

I implemented the routing as `topk(logits)` → `softmax(top-k values)`,
which is mathematically equivalent to `softmax(all logits)` → `topk`
→ renormalize (for the `norm_topk_prob=True` case). Because softmax
is monotonic, top-k indices are the same either way; and:

$$
\mathrm{softmax}(\mathrm{topk}(l))_i = \frac{e^{l_i}}{\sum_{j \in \text{topk}} e^{l_j}} = \frac{e^{l_i}/Z}{\sum_{j \in \text{topk}} e^{l_j}/Z} = \frac{\mathrm{softmax}(l)_i}{\sum_{j \in \text{topk}} \mathrm{softmax}(l)_j}
$$

The $Z$ (full-sum normalization) cancels. So `softmax` over just the
top-k logits gives the same weights as full-softmax renormalized over
top-k. This saves the full-softmax over $E$ (= 128 for Qwen3-30B-A3B)
when top_k=8. Only the `norm_topk_prob=True` branch is equivalent; the
un-normalized branch keeps the full softmax and gathers.

Numerical caveat: full-softmax should still be **fp32-cast then back**
for stability. My initial version skipped this and passed anyway
because $E=8$ in the test config leaves plenty of bf16 headroom; at
Qwen3 scale (E=128) it would matter.

### 3. Per-expert loop pattern

```python
for e in range(num_experts):
    token_idx, k_idx = torch.where(selected_experts == e)      # both [num_tokens_for_e]
    if token_idx.numel() == 0: continue
    expert_out = self.experts[e](x_flat[token_idx])            # [t, H]
    expert_out = expert_out * routing_weights[token_idx, k_idx, None]
    y_flat.index_add_(0, token_idx, expert_out)
```

The elegance: **loop over experts, not over tokens**. Each iteration
gathers the (typically few) tokens routed to expert $e$, runs one
forward on the mini-batch, weights by that token's routing prob for
$e$, scatter-adds into the output. Poor GEMM efficiency because each
expert sees $N \cdot k / E$ tokens on average (small tensor,
kernel-launch dominated) — Ex05b's permuted layout is the
grouped-compute fix.

### 4. PyTorch mechanics consolidated in this exercise

Not new, but the ones that showed up together and are worth having as
a mental checklist:

- **`torch.where(cond)` returns a tuple of coordinate tensors**, one
  per axis of `cond`. `token_idx, k_idx = torch.where(selected_experts == e)`
  destructures a 2-D condition into two 1-D index tensors. Same idea
  as `numpy.nonzero`, different name.
- **`torch.topk(x, k, dim)` returns `(values, indices)`** — same
  shape signature as `torch.max` (`torch.max` is just `topk(k=1)`
  under the hood). Both `values` and `indices` have shape
  `(..., k)` (the reduced dim replaced with $k$).
- **`x[..., None]` = `.unsqueeze(-1)` in Triton and NumPy dialect.**
  Extra axis for broadcasting; here used to align 1-D routing weights
  `[t]` with 2-D expert output `[t, H]` for elementwise multiply.
- **`index_add_(dim, index, source)`** is the scatter-add primitive.
  Not deterministic on CUDA by default — multiple threads can race on
  the same output row. For our tests the tolerance handles it; for
  the paper's determinism argument we'll need
  `torch.use_deterministic_algorithms(True)` (or a deterministic
  reduction kernel).
- **`.view()` vs `.reshape()`**: `.view()` requires contiguous;
  `.reshape()` handles non-contiguous silently by copying. For
  `y_flat = torch.zeros_like(x_flat)` the input is contiguous so
  either works; `.reshape()` is more robust to code churn.

### 5. `nn.ModuleList` vs plain Python list — the real reason

Long side-discussion, worth codifying. Plain list in place of
`nn.ModuleList([SwiGLU_MLP(...) for _ in range(E)])` has four failure
modes, three of which matter for inference-only workloads too:

| Failure mode | Training? | Inference? |
|---|---|---|
| Optimizer skips params | Yes | N/A |
| Gradient tracking off | Yes | N/A |
| `.to(device, dtype)` skips experts | Yes | **Yes** — 1st forward errors |
| `state_dict()` / `load_state_dict()` skips experts | Yes | **Yes** — random-init experts |
| `torch.compile` graph capture incomplete | Yes | **Yes** — fusion misses |
| `sum(p.numel() for p in .parameters())` under-reports | Yes | **Yes** — memory sizing wrong |

The only situations plain list works for inference: construct on
target device via `set_default_device` AND load weights via direct
attribute access (bypass `state_dict`) AND don't use `torch.compile`
AND don't do introspection. Narrow set of conditions.

Root cause is Python's `nn.Module.__setattr__` **only registers
`nn.Module` and `nn.Parameter` values by name**. Assigning a plain
list of `nn.Module` instances stores the list itself, and the list is
not a `Module` — so the framework can't see through it during
`.parameters()` / `.to()` / `.state_dict()` walks. `nn.ModuleList` IS
an `nn.Module` subclass, so it registers, and its internal `.append`
does the per-child `_modules[i] = child` bookkeeping. Same trick
`nn.ModuleDict` and `nn.ParameterList` use.

### 6. Traps I hit (1 bug)

- **`nn.Linear(hidden, num_experts)` missing `bias=False`** on the
  router. `nn.Linear` defaults to `bias=True`. The test's
  `_copy_weights` only copies `.weight`, so the un-copied random
  `bias ~ U[-1/\sqrt{H}, 1/\sqrt{H}] \approx \pm 0.02` corrupts
  routing. Test failed until fixed. `RefSparseMoE` uses `bias=False`
  (matches HF/vLLM convention — routers have no additive bias in
  every production MoE codebase). **Pair every MoE router `nn.Linear`
  with `bias=False`** — mnemonic: "router is a scoring function, not
  an affine map."

### 7. Verification framing

- **Ex05a's forward is single-GPU, no comm.** So the correctness
  spec is just "output matches per-token weighted sum of top-k
  expert outputs" — same functional predicate as
  `RefSparseMoE.forward`. No collective invariants yet.
- **The permutation vocabulary starts here** in a rudimentary form:
  `torch.where(selected_experts == e)` is a *partial* permutation
  (extract one expert's tokens at a time). Ex05b will build the
  **full** permutation (argsort by expert → contiguous blocks) that
  Ex06's all-to-all dispatch operates on.
- **The routing function's post-condition** is worth stating
  precisely, because it's the interface between Ex05a's naive path
  and Ex05b's permuted path (and Ex06's EP path):

  For each token $i$, produce $(e_{i,1..k}, w_{i,1..k})$ such that:
  1. $e_{ij} \in \{0, \dots, E-1\}$, all distinct within a token
     (top-k gives distinct indices);
  2. $\sum_j w_{ij} = 1$ if `norm_topk_prob`, else $\sum_j w_{ij} \le 1$;
  3. $w_{ij} \ge 0$ (softmax outputs are non-negative).

  Any downstream MoE variant that satisfies these three properties is
  interchangeable at the routing interface. Ex05b's permuted layout
  gives the same $(e, w)$ but reorganizes the physical layout; Ex06's
  EP splits the compute across ranks. Same routing spec, three
  physical realizations.

### Reference material worth revisiting

- HF's [`Qwen3MoeSparseMoeBlock`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) — the naive per-expert loop shipped in production.
- [nanovllm-jun `Qwen3SparseMoeBlock._forward_expert_parallel`](../nanovllm-jun/nanovllm/models/qwen3.py) — the EP variant. Uses the permuted layout Ex05b introduces, plus all-to-all comm from Ex06.
- Shazeer et al. 2017 ["Outrageously Large Neural Networks"](https://arxiv.org/abs/1701.06538) — the paper that introduced Sparsely-Gated MoE. Router + top-k + weighted combine is theirs.
- Fedus et al. 2021 ["Switch Transformer"](https://arxiv.org/abs/2101.03961) — top-1 special case, popularized MoE for large-scale LLM training.

## Ex05b — Permuted Sparse MoE (single-GPU, grouped compute)

**Status: done, 2/2 tests green.** Same math as Ex05a; different data
layout. Tokens get **permuted into per-expert contiguous blocks** so
each expert's compute is a contiguous-slice matmul instead of a
fancy-index gather. Introduces the argsort/bincount/offsets vocabulary
that every downstream MoE step (Ex06 EP, Ex09 fused kernel) rides on.

### 1. The permutation vocabulary — 7 steps

Given routing output `top_k_experts: [N, top_k]` and `top_k_weights: [N, top_k]`:

| Step | Operation | Shape produced |
|---|---|---|
| 1 | Router (same as Ex05a) | `top_k_experts`, `top_k_weights` [N, k] |
| 2 | Flatten to (N·k) records | `token_ids_flat`, `expert_ids_flat`, `weights_flat` [Nk] |
| 3 | Argsort by expert (`torch.sort` returns values + perm indices) | `sort_perm` [Nk] |
| 4 | Bincount + cumsum → offsets | `offsets` [E+1] |
| 5 | Gather sorted input via fancy indexing | `sorted_x` [Nk, H] |
| 6 | Per-expert compute on contiguous slice `sorted_x[offsets[e]:offsets[e+1]]` | `sorted_out` [Nk, H] |
| 7 | Weight + unpermute + `index_add_` back to [N, H] | `y_flat` [N, H] |

Steps 3–5 are the "permutation dance." Steps 6–7 are the compute +
combine. Ex06 will insert `all_to_all_variable` between steps 5 and 6
(dispatch) and again between steps 6 and 7 (combine). Ex09 will
replace step 6 with a single Triton kernel launch.

### 2. Naive vs argsort — same math, three implementation forms

A `for expert_id in range(E): torch.where(top_k_experts == expert_id)`
loop **already produces the same per-expert grouping** as an explicit
argsort — the loop order IS the permutation. For single-GPU Ex05b in
isolation, either form gives the same output.

**What actually differs is materialization discipline** — how much of
the pipeline lives outside the expert-loop body. My Ex05b went through
three variants of increasing lift-out:

| Form | Loop body does | Outside the loop |
|---|---|---|
| **V1: `torch.where` per iter** (my first draft) | `where` + gather + mul + scatter + expert forward | argsort + bincount only |
| **V2: `x_flat_permuted` upfront** (after 1st refactor) | slice + mul + scatter + expert forward | argsort + gather + bincount + weights permute |
| **V3: full reference form** (final) | slice + expert forward + slice-write | argsort + gather + expert-output buffer + one final `expert_output *= sorted_weights[:, None]` + one final `y_flat.index_add_(...)` |

Corrected launch-count comparison (accounting for E `copy_` launches
in V3's `expert_output[s:e] = self.experts[e](...)` slice-writes,
which I initially undercounted):

| Form | Non-expert launches (E=128) | Total launches (E=128, incl. ~5E from expert bodies) |
|---|---|---|
| V1 (`torch.where` per iter) | ~3E = 384 | ~8E = 1024 |
| V2 (`x_flat_permuted` upfront) | ~2E + 1 = 257 | ~7E = 897 |
| V3 (full reference form) | ~E + 3 = 131 | ~6E = 771 |

**V3 is ~25% fewer launches than V1, not the ~100× ratio my initial
writeup claimed.** The real advantage of V3 over V1/V2 isn't launch
count; it's **memory-access pattern + Ex06 readiness**:

| Concern | V1 | V2 | V3 |
|---|---|---|---|
| Expert forward reads via | fancy-index gather (strided HBM) | contiguous slice view | contiguous slice view |
| Materialized `[Nk, H]` input buffer | No | Yes (`x_flat_permuted`) | Yes + `expert_output` |
| `all_to_all_variable` (Ex06) input | Needs added gather | Ready | Ready |
| Ex09 fused-kernel input format | Needs restructuring | Compatible | Compatible |
| Loop body reads/writes | scattered | contiguous read, per-iter scatter | contiguous read, contiguous write |

**Argsort + full-lift-out is preparation, not necessity.** Ex05b's job
is to install the vocabulary + data layout that Ex06 and Ex08 require.
The mnemonic: "same math, three physical realizations" —
naive/permuted-partial/permuted-reference all satisfy the same routing
spec; only the memory-access shape and Ex06/Ex08 compatibility differ.

### 3. Memory cost of the permuted layout

`sorted_x` and `sorted_out` are each `[N·top_k, H]` — a **factor-k
blowup** on the activation tensor. At Qwen3-30B-A3B (N≈1024, k=8,
H=2048, bf16): ~32 MB per tensor per MoE layer, peak ~64 MB
per-layer transient. Trivial on 80 GB A100.

**Not a new cost, just re-shaped**: Ex05a's naive loop also allocates
E small `x[token_idx]` tensors totaling the same bytes; the difference
is peak vs cumulative allocation. Ex05b trades higher peak for lower
allocator churn (2 slabs vs E small tensors).

**The blowup is unavoidable once EP is on the table**: `all_to_all_variable`
in Ex06 requires one contiguous send buffer of shape `[Nk, H]`. Ex05b
pays this memory cost one exercise early; from there it's monotone
through Ex06/Ex08 with no undo.

Top-1 MoE (Switch Transformer) escapes this — top_k=1 means no blowup.
Top-k MoE (Qwen3, Mixtral, DeepSeek, etc.) always pays the k× cost.

### 4. The "three tensors, one permutation" rule

In the permuted layout, three parallel 1-D tensors travel in
lockstep, all keyed on the same axis `p ∈ [0, N·top_k)`:

| Tensor | Semantics |
|---|---|
| `sorted_token_ids[p]` | which token this record refers to |
| `sorted_expert_ids[p]` | which expert receives this record |
| `sorted_weights[p]` | routing weight for this (token, expert) pair |

**Every one of them must get permuted by the SAME `sort_perm`.**
Skipping the permutation on any of the three means the correspondence
between positions breaks silently — and my Ex05b bug (§6 below) hit
exactly that failure mode. `argsort` returns an index tensor
specifically so you can apply the same permutation to multiple
tensors — this is what makes gather/scatter reusable across
downstream steps.

### 5. PyTorch mechanics consolidated in this exercise

- **Fancy indexing shape rule**: `x[idx].shape == idx.shape + x.shape[1:]`.
  Output leading shape follows the index tensor's shape, not the
  source's. So `x[sorted_token_ids]` where `sorted_token_ids: [Nk]`
  gives `[Nk, H]` output — even though the source `x` has only N
  rows. `max(idx) < len(x)` is all that's required for validity.
- **Reads with duplicates are cheap and race-free** — each output
  thread reads one source row, no cross-thread contention. **Writes
  with duplicates race** and need `index_add_` (which uses atomics
  on CUDA).
- **Fancy indexing → copy; slice indexing → view.**
  `x_flat[sorted_token_ids]` allocates a new tensor;
  `sorted_x[offsets[e]:offsets[e+1]]` is a zero-copy view. Ex05b uses
  both.
- **`torch.sort(x)` returns `(values, indices)`** — same info as
  `torch.argsort` but the values come along, saving a re-gather. Use
  `stable=True` for reproducibility across runs (default is False
  in torch < 2.4).
- **`torch.bincount(x, minlength=E)`** — the `minlength` argument is
  mandatory when the tail experts may be empty. Without it, output
  shape is `[max(x) + 1]`, not `[E]`.
- **`F.pad(cumsum(counts), (1, 0))`** is the cleanest prepend-zero
  idiom for the offsets tensor. Alternatives (`torch.cat([zeros(1),
  ...])`) work but are more verbose.
- **`torch.topk(..., sorted=True)` (the default) makes indices come
  out in descending-weight order** — irrelevant for correctness
  (subsequent argsort re-orders them), but a determinism gift for the
  paper: the tie-breaking is stable across runs without extra effort.

### 6. Traps I hit (2 bugs)

- **`nn.Linear(hidden, num_experts)` missing `bias=False` on the
  gate** — literal repeat of the Ex05a bug. Same failure mode
  (`_copy_weights` copies only `.weight`, un-copied random bias
  corrupts routing). Fix: pair every MoE router with `bias=False`
  by reflex. Every production MoE codebase does this — no exceptions.
- **Weight lookup used the wrong index space.** After building
  `token_idx_rep_permuted_by_experts` (token IDs in expert-grouped
  order), I indexed `top_k_weights_flat` with those token IDs to get
  the weights for each expert's tokens. Wrong index space:
  `top_k_weights_flat` is keyed by **flat routing record position
  `p ∈ [0, Nk)`**, not by token ID `t ∈ [0, N)`. Passing token IDs
  fetched arbitrary weights (routing records at positions matching
  the token-ID numeric values), silently producing wrong output that
  matched shape but not values. Fix: build a `top_k_weights_flat_permuted`
  tensor once (`top_k_weights_flat[top_k_experts_permutation]`) and
  slice IT via `[start : start+token_cnt]` inside the loop — symmetric
  with how `token_idx_rep_permuted_by_experts` is used. This is the
  "three tensors, one permutation" rule violated.

### 7. Verification framing

**Ex05b's abstract post-condition is identical to Ex05a's** — same
three routing invariants (distinct indices per token, non-negative
weights, sum-to-1 under `norm_topk_prob`), same weighted-combine
result per token. **Both implementations refine the same abstract
spec**: they compute the same $Y_i = \sum_j w_{ij} f_{e_{ij}}(X_i)$
up to fp reduction-order.

For Verus/Dafny: this is a bisimulation-type observation. Two
concrete implementations satisfying the same abstract post-condition
are interchangeable at the paper's spec level. The permutation π
that groups by expert-id is quantified existentially in the abstract
model; both naive-loop-order and argsort realize it. **Same proof
covers both implementations without change.** This is a nice payoff
of picking a mostly-declarative spec — the physical layout is below
the abstraction level.

The interesting NEW spec content in Ex05b, compared to Ex05a:

- **Permutation invariants**: `sort_perm` is a bijection on
  `[0, N·top_k)`. Applied to `expert_ids_flat`, the result is
  sorted (monotone non-decreasing). This is a proof of concept for
  the argsort-permutation-schema that Ex06 will inherit.
- **Offset well-formedness**: `offsets[E+1]` is a strictly
  monotone-non-decreasing sequence with `offsets[0] = 0` and
  `offsets[E] = N·top_k`. Any block-based iteration over the sorted
  layout obeys `sum of block lengths = total records`. Bincount +
  cumsum + prepend-zero produces exactly this shape.
- **Scatter-add commutativity**: `index_add_(sorted_token_ids, ...)`
  accumulates each token's `top_k` contributions in some order. The
  abstract sum is commutative, so any order gives the same result up
  to fp reduction-order. **This is the one non-trivial fp-associativity
  bookkeeping** the paper will need to state precisely (tolerance is
  ~1e-2 in bf16 for 128-expert × top_k=8).

### 8. Looking ahead — no atomics for Ex09 fused kernel

The Triton kernel in Ex09 (grouped MoE GEMM) is **atomics-free** —
structurally identical to FA2 forward. Each expert's block
`[offsets[e]:offsets[e+1]]` is a disjoint tile of the output; no two
kernel blocks write to the same output row. All reduction
(matmul-inner-sum) is intra-block via shared memory. `tl.atomic_*`
only appears if step 7 (the scatter-add) is fused INTO the same
kernel — most production kernels (Megablocks, vLLM `FusedMoE`) keep
step 7 as PyTorch's `index_add_` outside the kernel, avoiding
in-kernel atomics entirely. For the paper's verification story this
is a clean choice: `index_add_` is a well-defined abstract collective
event; in-kernel atomics would require weak-memory reasoning inside
Triton.

### 9. Materialization discipline — why V3 is Ex06-ready

The V1 → V3 refactor path is worth codifying because it's the
**exact structural progression** Ex06 will demand. In V3, the code
has a clean three-block shape:

```python
# Block A: LOCAL prep (router + argsort + prep buffers)
router_logits = self.gate(x_flat)
top_k_weights, top_k_experts = torch.topk(...)
...
x_flat_permuted = x_flat[token_idx_rep_permuted_by_experts]      # [Nk, H]
top_k_weights_flat_permuted = top_k_weights_flat[permutation]     # [Nk]
expert_output = torch.zeros_like(x_flat_permuted)                 # [Nk, H]

# Block B: LOCAL expert loop (compute-only, no weight math, no scatter)
start = 0
for expert_id in range(self.num_experts):
    ...
    expert_output[start:start+cnt] = self.experts[expert_id](x_flat_permuted[start:start+cnt])
    start += cnt

# Block C: LOCAL combine (one big multiply + one big scatter)
expert_output *= top_k_weights_flat_permuted[:, None]
y_flat.index_add_(0, token_idx_rep_permuted_by_experts, expert_output)
```

**Ex06 slots in with minimal surgery**: two `all_to_all_variable`
collectives wrap Block B, and Block B's loop range narrows from
`num_experts` to `experts_per_rank`. Nothing else changes.

```python
# Block A: same as V3.

# NEW: DISPATCH (all_to_all_variable) — send my sorted_x to expert-owning ranks
input_splits, output_splits = compute_splits(top_k_experts_ids, self.experts_per_rank)
local_x = all_to_all_variable(x_flat_permuted, input_splits, output_splits)
local_expert_output = torch.empty_like(local_x)

# Block B, MODIFIED: local loop over only my experts (E/EP iterations)
for local_e in range(self.experts_per_rank):
    ...
    local_expert_output[s:e] = self.experts[local_e](local_x[s:e])

# NEW: COMBINE (all_to_all_variable) — reverse dispatch, results back to originating rank
expert_output = all_to_all_variable(local_expert_output, output_splits, input_splits)

# Block C: unchanged — multiply-by-weights and scatter on ORIGINATING side, no cross-rank comm.
expert_output *= top_k_weights_flat_permuted[:, None]
y_flat.index_add_(0, token_idx_rep_permuted_by_experts, expert_output)
```

**Six new lines, one line changed.** That's the payoff of the
materialization discipline in V3.

**The key semantic observation**: Block C stays on the originating
rank because `sorted_token_ids` and `sorted_weights` never leave.
The compute rank (which owns some experts) doesn't have those
tensors — they're purely local to the token's originator. This
mirrors the paper's abstract framing: an expert is a **pure function
$R^H \to R^H$**; routing weights are a property of the (token, router)
interaction, not (token, expert). Dispatch and combine carry only
hidden states; weights and token IDs stay put. Slimmer collective
payloads, cleaner Verus spec.

**Why V1 doesn't extend to Ex06 cleanly**: without a materialized
`[Nk, H]` buffer, there's nothing to `all_to_all_variable` — the
collective wants one contiguous send buffer, not a stream of per-expert
tiny tensors. V1 → V3 isn't a perf-only refactor; it's a **prerequisite
for the very-next-exercise's collectives**.

### Reference material worth revisiting

- Fedus et al. 2021 ["Switch Transformer"](https://arxiv.org/abs/2101.03961) — top-k=1 case; no factor-k blowup, but same permutation pattern in an EP setting.
- Megablocks ["Efficient MoE Training" (Gale et al. 2022)](https://arxiv.org/abs/2211.15841) — sparse MoE kernels with the same permutation vocabulary; different naming (`bin_ids`, `bin_offsets`).
- vLLM `FusedMoE` in [`vllm/model_executor/layers/fused_moe/`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) — the production Triton kernel implementation. `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded` naming.
- [nanovllm-jun `Qwen3SparseMoeBlock._forward_expert_parallel`](../nanovllm-jun/nanovllm/models/qwen3.py) — the EP variant our bootcamp is preparing us to fix. Uses the same permutation vocabulary + `all_to_all_variable`.

---

## Cross-cutting: Ecosystem naming conventions (MoE)

Every MoE codebase reinvents slightly different names for the same
concepts. Below is my translation table between the bootcamp's
naming (chosen to match my own mental model, matching the
[README.md](ex05_moe_baseline/README.md) math notation where
possible), HF Transformers, and vLLM / nanovllm-jun. Use this as the
port-back document when moving bootcamp code into nanovllm-jun.

| Concept | Bootcamp (mine) | HF Transformers | vLLM / nanovllm-jun |
|---|---|---|---|
| Router pre-activation | `router_logits` | `router_logits` | `router_logits` |
| Full softmax over experts | `routing_probs` (mid-computation, discarded) | `routing_weights` | `router_probs` |
| Post-top-k weights | `top_k_weights` | `routing_weights` (reused name) | `topk_weights` |
| Post-top-k expert indices | `top_k_experts` | `selected_experts` | `topk_ids` |
| Renormalize top-k flag | `norm_topk_prob` | `norm_topk_prob` (config) | `renormalize` (constructor arg) |
| Router linear layer | `self.gate` | `self.gate` | `self.gate` (as `ReplicatedLinear`) |
| Experts container | `self.experts: nn.ModuleList` | `self.experts: nn.ModuleList` | `self.experts` (single `FusedMoE` object, or `ModuleList` in nanovllm-jun) |
| Full flat expert-id tensor | `top_k_experts_flat` (or `expert_ids_flat`) | *(N/A — HF doesn't permute)* | `expert_ids` (in kernel signatures) |
| Argsort permutation index | `top_k_experts_permutation` (or `sort_perm`) | *(N/A)* | `sorted_indices` / `permutation` |
| Post-sort expert IDs | `top_k_experts_ids` | *(N/A)* | `sorted_expert_ids` |
| Post-sort token IDs | `token_idx_rep_permuted_by_experts` (or `sorted_token_ids`) | *(N/A)* | `sorted_token_ids` |
| Post-sort routing weights | `top_k_weights_flat_permuted` (or `sorted_weights`) | *(N/A)* | `sorted_weights` |
| Bincount by expert | `expert_bincnt` | *(N/A)* | `num_tokens_per_expert` |
| Cumulative offsets | `offsets` (or start-position tracked in loop) | *(N/A)* | `expert_offsets` / `token_offsets` (Megablocks: `bin_offsets`) |
| Permuted input tensor `[N·k, H]` | `sorted_x` | *(N/A)* | `permuted_hidden_states` / `sorted_hidden_states` |
| Post-scatter final output | `y_flat` | (implicit in HF's naive combine loop) | `hidden_states` (overwritten) |

**HF is the "readable tutorial" naming** — descriptive, close to
math notation. **vLLM / nanovllm-jun is the "kernel-friendly"
naming** — shorter tokens that match Triton argument names.
**Bootcamp is somewhere in between**, biased toward descriptive
names where they line up with the paper's math.

**When porting**: the port from bootcamp to nanovllm-jun is a
mechanical search-and-replace against this table, roughly a
one-commit refactor. Don't rename now — rename when actually porting.

---

## Cross-cutting: Sparse MoE as Strang's 4th way of matmul

An abstract framing I want to keep at hand — the "loop over experts,
not tokens" reordering that felt like a clever data-path trick in
Ex05a is actually **Strang's 4th way of matrix multiplication**
applied to a sparse rank decomposition. Same idea as PCA, LoRA, and
tiled GEMM. Naming this precisely helps position the paper.

### Recall — Strang's four ways to compute $C = AB$

| Way | Formula | Iteration |
|---|---|---|
| 1. Inner products | $C_{ij} = \sum_k A_{ik} B_{kj}$ | (i, j) outer, k inner |
| 2. Column-by-column | $C_{:,j} = A \cdot B_{:,j}$ | j outer |
| 3. Row-by-row | $C_{i,:} = A_{i,:} \cdot B$ | i outer |
| **4. Sum of rank-1 outer products** | $C = \sum_{k} A_{:,k}\, B_{k,:}$ | **k outer** — sum over shared axis |

Way #4 is Strang's headline framing: matmul as accumulation of
rank-1 contributions, one per shared-axis index.

### MoE combine as way #4

The MoE combine is a matmul-like contraction over the expert axis:

$$
Y = \sum_{e=0}^{E-1} R_{:,e} \odot f_e(X)
$$

Where:
- $R \in \mathbb{R}^{N \times E}$ is the (sparse) routing matrix,
  $R_{ie} = w_{ij}$ if $e_{ij} = e$ else 0 — only top_k non-zeros per row.
- $f_e(X) \in \mathbb{R}^{N \times H}$ is expert $e$'s output on all
  tokens (a nonlinear function, so not quite "matmul" — but the
  outer-sum structure is identical).
- The sum ranges over the expert axis $e$ — Strang's shared axis.

Each term $R_{:,e} \odot f_e(X)$ is a **rank-slice-in-e contribution**
to $Y$. Sum over $E$ terms → total output. Structurally way #4.

**The per-expert loop physically realizes this decomposition.** Each
iteration computes one term of the sum. The sparsity of $R$ (only
top_k non-zeros per row) means you compute $f_e$ only for the tokens
routing to $e$ — a **sparse rank-slice**, not a full one — but the
outer-decomposition structure is intact.

**The per-token loop (the "obvious" alternative that Ex05a rejects)**
would be way #3 (row-by-row): compute $Y_i$ for each token
independently by iterating its top_k experts. Terrible GEMM shape —
Nk tiny matmuls, one per (token, expert) pair, no batching. Way #4
pools tokens by shared expert, giving each iteration a proper
`[t_e, H] @ [H, I]` batch.

### PCA connection

Truncated PCA:

$$
X \approx \sum_{k=1}^{K} \sigma_k u_k v_k^\top
$$

Truncated sum of rank-1 outer products; top-K captures the "important"
contributions.

Sparse MoE:

$$
Y = \sum_{e \in \text{active}} R_{:,e} \odot f_e(X)
$$

Truncated sum of rank-slice contributions; **top_k routing captures the
"important" experts per token**.

| PCA | MoE |
|---|---|
| Basis $u_k, v_k$ | Expert function $f_e$ + column of $R$ |
| Weight $\sigma_k$ | Routing weight $w_{ie}$ |
| Truncate to top-K by $\sigma$ | Truncate to top-k by softmax |
| Sum contribution = $\sigma_k u_k v_k^\top$ | Sum contribution = $w_{ie} \cdot f_e(X_i)$ |

**Both are top-K truncations of a rank-decomposed sum.** Different
"basis" (linear singular vectors vs. nonlinear expert functions),
same structural pattern.

The truncation is why sparse-MoE is efficient. **Without top-k, MoE
would be a rank-E outer sum over all tokens — cost of $E$ MLPs per
token.** Top-k drops to $k \ll E$ active contributions, an $E/k$
speedup. That's the whole efficiency case for sparse MoE in one
observation.

### The bigger pattern — Strang's #4 shows up everywhere

The "reorder the summation axis" move is a fundamental HPC pattern:

| Where | The decomposition | Purpose |
|---|---|---|
| GEMM loop tiling (M-first vs K-first) | Reorder $\sum_k$ vs $\sum_i$ vs $\sum_j$ | Cache alignment |
| Sparse attention (Longformer / BigBird) | $Y = \sum_{\text{block}} A_{\text{block}} V_{\text{block}}$ | Skip empty blocks |
| Einsum reordering | Choose which axis to reduce first | Cost heuristics |
| Multi-head attention | Head axis outer: $Y = \text{concat}_h A_h V_h$ | Head-parallel independent compute |
| LoRA | $W = W_0 + BA$ (rank-r additive contribution) | Cheap fine-tuning |
| GPTQ / block-diagonal quantization | $W = \sum_b W_b$ (block sum) | Independent quantization per block |
| Tensor-train / low-rank compression | High-D tensor as sum of low-rank cores | Memory saving |

**Every one of these is way #4 applied to a specific structure.**
Sparse MoE is just: "way #4 over the expert axis, sparse in the
non-selected experts."

### Implication for parallelism

Once MoE is framed as Strang's #4 over the expert axis, the
parallelism-axis story becomes clean:

- **TP (Tensor Parallelism)** shards within each rank-slice — shards
  the *hidden* axis inside a single expert's compute.
- **EP (Expert Parallelism)** shards the outer sum axis — assigns
  different terms of $\sum_e$ to different ranks.

**TP and EP are orthogonal because they partition different axes of
the Strang-4 decomposition.** The paper's Ex07 composition theorem
is: partitioning distinct axes of a rank-decomposable sum preserves
correctness under fixed collective schedules. Verus/Dafny quantify
this as a Hoare-triple over the sum-decomposed spec.

### Implication for verification

The paper's abstract spec of the MoE forward is:

$$
Y \;=\; \bigoplus_{e=0}^{E-1} \bigl(R_{:,e} \odot f_e(X)\bigr)
$$

Where $\oplus$ is the abstract "sum" collective (commutative,
associative up to fp reduction-order tolerance). Every physical
realization — Ex05a naive, Ex05b permuted, Ex06 EP, Ex07 TP+EP, Ex08 weiz-schedule,
Ex09 fused kernel — refines this spec by choosing:

1. **The order of the sum** (per-expert iteration order).
2. **The physical layout** ($[N, H]$ vs $[Nk, H]$ intermediate).
3. **The parallelism partition** (which ranks own which $e$).
4. **The kernel fusion boundary** (which sums live inside one kernel launch).

The correctness theorem in each case is: "this physical schedule
computes the abstract Strang-4 sum, up to declared reduction-order
tolerance." **Six exercises, one theorem shape.** Clean composition
point for the paper.

### The one-line takeaway

**Sparse MoE = Strang's 4th way of matmul, truncated by top-k routing,
with expert functions replacing rank-1 outer products.** TP shards
within a term; EP shards across terms; the sum is preserved.

---

## Ex06 — Expert Parallelism (only pure variant implemented)

**Status: `ex06_ep_pure/` implemented and green. `ex06_ep/`
(canonical and lean) kept as reference-only records of a design
misstep — never implemented by me.** This exercise taught more than
the original plan intended: I designed a variant, questioned the
framing, discovered the design conflated TP-scope with EP-scope
properties, rebuilt in `ex06_ep_pure/` under the correct framing,
and left the mis-framed `ex06_ep/` as a documented record. The
"replicated-input EP" pattern that motivated `ex06_ep/` isn't a
natural pure-EP shape — it's what a TP layer produces upstream —
and it belongs in Ex07's TP+EP hybrid, not here.

The three files that exist under `ex06_ep/`:

| Variant | Directory | Input pre-condition | Collectives per forward | Status |
|---|---|---|---|---|
| Canonical | `ex06_ep/reference.py` | Replicated across ep_group | 5 (splits + 2 dispatch + combine + all_gather) | Reference only — no solution written |
| Lean | `ex06_ep/reference_lean.py` | Replicated across ep_group | 1 (all_reduce) | Reference only — no solution written |
| **Pure** | **`ex06_ep_pure/`** | **Distinct per rank (DP-style)** | **4 (splits + 2 dispatch + combine)** | **Implemented, 8/8 green** |

**Why `ex06_ep/` is reference-only**: its framing ("input replicated
across ep_group") is unnatural for a pure EP exercise — it imports a
TP-scope property (attention's row-parallel AR produces replicated
hidden states) and disguises it as an EP-scope invariant. When the
replicated-input question surfaces in `ex06_ep/`, the exercise
becomes "why doesn't lean-style all_reduce just replace dispatch?"
which is a valid question but a **TP+EP composition** question, not
a pure-EP algorithm question. Ex07 will re-encounter the same
question in its proper context — where TP is actually providing the
replication and the design trade-off is meaningful.

The `ex06_ep/` code and benchmarks are preserved as a record of the
"nanovllm-jun-inherited redundant-dispatch pattern" for the paper's
port-back discussion, but pedagogically **the pure-EP variant in
`ex06_ep_pure/` is what teaches the EP algorithm.**

### 1. The three input regimes and what they demand

The exercise revealed a design conflation between **TP-scope** and
**EP-scope** properties that I hadn't seen before:

| Setup | Input across ep_group | Dispatch necessary? | Lean shortcut? |
|---|---|---|---|
| **Pure EP** (DP-partitioned, `ex06_ep_pure/`) | Distinct per rank | **Yes** | No — non-owners have no data |
| **TP == EP == world** (`ex06_ep/`, nanovllm-jun's actual topology) | Replicated (from attn AR) | **No — redundant** | Yes — `all_reduce` alone suffices |
| DP × TP × EP hybrid (Ex07 territory) | Replicated within TP subgroups | Partially — only for cross-DP-replica traffic | Only within TP scope |

**Input replication is a TP-scope property.** It arises when attention's
row-parallel all-reduce covers the same rank set as EP. Under that
overlap, the dispatch/combine pipeline moves data that's already at the
destination — redundant work inherited from training-scale MoE
patterns where inputs ARE partitioned by DP and dispatch is
load-bearing.

**Pure EP** without TP subsumption (each rank has distinct data) is
where dispatch actually earns its keep. `ex06_ep_pure/` teaches the
algorithm in this natural form; the delta from `ex06_ep/` is just
"remove the redundant input-slice and final all_gather wrapper."

### 2. The dispatch/combine algorithm (canonical form)

The core algorithm shared by all three variants (with different
wrappers):

```
Phase 1: local router on local tokens                    → [local_N, top_k]
Phase 2: argsort by expert_id, gather permuted x         → sorted_x [local_N·k, H]
Phase 3: per-destination-rank splits via bincount        → input_split_sizes [ep_size]
Phase 4: all_to_all_single on splits                     → output_split_sizes [ep_size]
Phase 5: DISPATCH via all_to_all_variable × 2            → received_x, received_expert_ids
Phase 6: local re-argsort by local expert_id             → contiguous per-expert slices
Phase 7: local expert compute (loop over local experts)  → local_expert_out
Phase 8: reverse the local sort                          → unsorted_received_out
Phase 9: COMBINE via all_to_all_variable (SWAP splits!)  → returned_out
Phase 10: weight multiply + local scatter_add            → local_y_flat
Phase 11: [canonical only] all_gather to full replicated → y_flat [N, H]
```

The lean variant replaces phases 3-11 with a single `all_reduce` on a
partial output tensor. Pure variant keeps phases 1-10 and skips
phase 11 (no reason to reassemble globally when each rank owns its
own share).

### 3. The paper's insight: dispatch is a training-code inheritance

Under `tp_size == ep_size == world_size` (vLLM / nanovllm-jun default
for Qwen3-30B-A3B), the dispatch collective moves tokens across the
network to ranks that ALREADY HAVE those tokens (from the TP-replicated
attention output). Each rank could instead **filter its own copy** to
just the tokens routing to its local experts and skip dispatch
entirely.

**Bandwidth accounting** (per rank per forward, Qwen3 config ep=8, top_k=8):

| Approach | Non-body wire traffic | Total collectives |
|---|---|---|
| nanovllm-jun (striped dispatch + all_reduce) | ~3.75·N·H | 6 |
| Canonical (`ex06_ep/reference.py`) | ~4·N·H | 5 |
| **Lean (`ex06_ep/reference_lean.py`)** | **~1.75·N·H** | **1** |

The lean variant is ~2× less bandwidth AND ~5× fewer collective
launches than what nanovllm-jun ships.

**Empirical measurements** on 8×A100 (single node, NVLink):

| Regime | Speedup (dispatch → lean) |
|---|---|
| ep=8, uniform routing, N=512-16K | 1.09-1.26× |
| ep=8, adversarial skew (all top-k on rank 0), N ≤ 4K | 1.10-1.25× |
| ep=8, adversarial skew, N ≥ 8K | 0.95-0.97× (dispatch wins slightly — data-aware) |
| **ep=16 cross-node RoCE**, uniform | **1.64-3.50×** |
| **ep=16 cross-node RoCE**, adversarial skew | **2.18-4.07×** |

**Cross-node is where lean dominates** — collective count savings
(1 vs 4-6) matter far more when each collective has ~ms-scale startup
on cross-node fabric. Measured cross-node bandwidth on our RoCE
25 GbE cluster: 2.7 GB/s asymptotic; typical production IB HDR is
~10× faster, which would narrow but not eliminate the lean advantage.

**Why nobody has published this**: MoE inference stacks (vLLM, SGLang,
nanovllm-jun) inherit dispatch from training-scale MoE (Shazeer 2017,
GShard, Megatron-DeepSpeed) where `ep_size ≫ top_k` and inputs ARE
DP-partitioned. Nobody re-examined the schedule for the specific
inference regime `tp_size == ep_size == world_size` where dispatch
becomes redundant. Historical inertia + code-uniformity between
training and inference + fine-grained-EP trends (ep ≥ 32) explain the
gap. The paper's contribution is naming this and quantifying the
delta.

### 4. Traps I hit in Ex06_ep_pure (2 bugs, both typos)

- **`local_weights = top_k_experts_flat[sorted_experts_perm]`** — copy-paste
  typo. Should be `top_k_weights_flat[sorted_experts_perm]`. The
  variable is intended to hold sorted routing weights (fractional 0-1)
  but was filled with sorted expert IDs (integers 0..num_experts-1).
  Phase 10's `Y *= local_weights[:, None]` then multiplied expert
  outputs by expert IDs, wrecking the output. Sneaky because the
  shape is correct — `[Nk]` either way. Only assert_close catches it.
- **Combine splits backwards**: called
  `all_to_all_variable(Y_t, input_split_sizes=send_cnts_lst, output_split_sizes=recv_cnts_lst)`
  when combine's send counts should be `recv_cnts_lst` and receive
  counts should be `send_cnts_lst`. My mental model was right ("reverse
  the dispatch") but I got trapped by naming — I read
  `input_split_sizes` as "the previous call's input" instead of "THIS
  call's send counts." Fixed by swapping the two args.

Both bugs are pure typos in that the algorithm was correct;
what tripped me up was **operation-local naming vs global-pipeline
naming** — torch's collective API uses the former, and my mental
carry-forward from dispatch used the latter.

**Rule I'm codifying**: every collective is self-contained. When
implementing "reverse-of-X," compute this call's send/recv counts
from scratch (from this call's perspective) — don't reuse variable
names from the forward direction.

### 5. Naming translation trap — recorded in Ex00 §7

The bug 2 confusion is documented in [../learning_summary.md#7-collectives-as-functions](learning_summary.md#7-collectives-as-functions)
under Ex00 — the "collectives-as-functions vs collectives-as-message-passing"
distinction. Reading torch's `input_split_sizes` / `output_split_sizes`
as "MPI's sendcounts/recvcounts for THIS call" (not "the pipeline's
original counts") avoids this trap in future exercises.

### 6. Verification framing

Ex06's four property classes for the paper's Verus/Dafny spec:

- **Deadlock freedom**: fixed collective schedule per variant. Every
  rank issues the same sequence (splits negotiation → dispatch × 2 →
  compute → combine → [all_gather or nothing]) regardless of routing
  distribution. Standard fixed-schedule proof.
- **Data-race freedom**: `index_add_` on CUDA uses atomics
  (non-deterministic reduction order), abstracted as an atomic
  commutative sum event. `all_reduce`'s reduction order is similarly
  abstracted.
- **Functional correctness**: pure EP output matches single-GPU
  `RefSparseMoE` up to fp reduction-order noise (~1e-4 fp32, ~5e-2
  bf16). The abstract spec is the same Strang-4 sum
  `Y = Σ_e R_{:,e} ⊙ f_e(X)`; each variant refines it via a different
  collective schedule.
- **Load-balance / progress**: under maximum routing skew (all top_k
  experts on one rank), the fixed schedule doesn't deadlock; empty
  local experts get `continue` (progress-preserving). All_reduce is
  data-independent (identical bytes regardless of skew); dispatch is
  data-aware (concentrates traffic where data is). Both are correct
  under any skew; wall-clock differs.

### 7. Empirical benchmarks worth citing in the paper

Ran on `bootcamp/ex06_ep/microbenchmark.py`:

- 8×A100 single node (NVLink): 1.09-1.26× lean speedup uniform, mixed under skew.
- 16×A100 across 2 nodes (RoCE 25 GbE): **1.64-4.07×** lean speedup, monotone in N and independent of routing.
- Measured cross-node bandwidth: 2.7 GB/s asymptotic (matches 25 Gbps Ethernet).

Configs tested: `H=2048 I=768 E=128 top_k=8` (Qwen3-30B-A3B),
`N ∈ [512, 16384]`, `dtype=bf16`, uniform and adversarial-skew routing.

### Reference material worth revisiting

- [nanovllm-jun `Qwen3SparseMoeBlock._forward_expert_parallel`](../nanovllm-jun/nanovllm/models/qwen3.py#L308) — the reference for what "canonical" EP inference looks like in production. Note the 3 all_to_all + 1 all_reduce structure; this is what our `reference.py` and `reference_lean.py` compare against.
- vLLM's [`FusedMoE`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) — production EP for large-scale serving; uses fused Triton kernels around the dispatch/combine layout.
- Gale et al. 2022 [Megablocks](https://arxiv.org/abs/2211.15841) — alternative sparse-GEMM approach; sidesteps dispatch entirely via block-diagonal kernels.
- Rajbhandari et al. 2022 [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) — the reference training-scale EP paper that established the dispatch/combine pattern nanovllm-jun inherits.

## Ex07 — TP + EP hybrid (case-3 canonical schedule)

**Setup**: 8 GPUs, TP-4 × DP-2 × EP-8. Two TP groups {0,1,2,3} and {4,5,6,7};
one EP group over all 8 ranks. Each TP group processes a distinct half of
the batch (DP-2 outer); MoE dispatch crosses TP-group boundaries.

Rank 0's per-block collective sequence, all in fixed order:

  attn AR (tp_group_a)
  → moe splits negotiation (ep_group)
  → moe dispatch × 2 (ep_group)             — x records + expert_id records
  → moe combine (ep_group)                   — reverse dispatch
  → moe all_gather (tp_group_a)              — un-stripe within TP

Every rank issues this exact same sequence with its own tp_group handle.
The paper's composition theorem argument for this schedule: cross-cutting
sub-groups (rank participates in both `tp_group` AND `ep_group`) do not
deadlock under fixed-schedule + explicit-group discipline.

### 1. The 11-phase pipeline

The MoE forward under case-3 has strictly more machinery than Ex06_ep_pure
because it must reconcile two parallelism dimensions:

| Phase | Op | Scope |
|---|---|---|
| 0 | Stripe within TP: `local_x = x_flat[tp_rank * local_N : (tp_rank+1) * local_N]` | local |
| 1 | Local router on `local_x` (gate + topk + softmax) | local |
| 2 | Sort by global expert id | local |
| 3 | Per-EP-dest-rank splits: `bincount(sorted_ids // experts_per_rank)` | local |
| 4 | Negotiate output splits | **all_to_all_single (ep_group)** |
| 5 | Dispatch `sorted_x` + `sorted_expert_ids` | **all_to_all_variable × 2 (ep_group)** |
| 6 | Subtract `expert_start` + local re-sort by local expert id | local |
| 7 | Loop over `experts_per_rank` — compute | local |
| 8 | Un-sort into received-order buffer | local |
| 9 | Combine (reverse dispatch with swapped splits) | **all_to_all_variable (ep_group)** |
| 10 | Un-sort into original order + weight-multiply + `index_add_` into `[local_N, H]` | local |
| 11 | Un-stripe: `all_gather local_y_flat → y_flat` | **all_gather_into_tensor (tp_group)** |

**The critical scoping distinction from Ex06's variants**:
- Phases 4, 5, 9 use `ep_group` (world-spanning).
- Phase 11 uses `tp_group` (within-TP).
- `ex06_ep/reference.py` had all collectives on `ep_group` — that mis-scoped
  the gather, which structurally should be TP-local.

### 2. Block-level invariant

  { input replicated within tp_group }
    → HybridBlock forward
  { output replicated within tp_group }

Two RMSNorms live between the sub-layers (pre-norm):

  h = x + attn(rmsnorm(x))
  y = h + moe(rmsnorm(h))

RMSNorm's weight is replicated across TP (data-unaware, per-hidden-dim gain).
Zero TP proof obligation — the primitive is trivially replicated. This is
worth naming explicitly in the paper: some primitives get dispatched in one
line of the correctness argument; others (Linear TP, MoE dispatch/combine)
carry the actual proof budget.

### 3. Design decisions worth citing

**Why `tp_size × dp_size == ep_size == world_size` (not `× ep_size`)**:
under our design where EP spans all ranks, `ep_size` isn't an independent
factor — it IS `world_size`. So the constraint is a 2-way factorization
`tp × dp = world`. Three canonical instances:

| Case | tp | dp | ep | Best schedule |
|---|---|---|---|---|
| 1 | 8 | 1 | 8 | Ex06 lean (all_reduce) |
| 2 | 1 | 8 | 8 | Ex06_ep_pure (pure EP dispatch) |
| 3 | 4 | 2 | 8 | Ex07 canonical (stripe + dispatch + all_gather) |

Ex07's code handles all three cases correctly. Cases 1 and 2 are handled
sub-optimally by Ex07's schedule (case 1 wastes bytes with dispatch instead
of all_reduce; case 2 wastes work with striping when tp=1), but the output
is correct. Formally: only the case-3 schedule needs an independent
correctness proof; cases 1 and 2 follow as degenerate instances.

**Why generic parameterization matters for the paper**: Ex07's constructor
takes `tp_size, tp_rank, tp_group, ep_size, ep_rank, ep_group` as generic
args and only asserts `num_experts % ep_size == 0` and `N_tp % tp_size == 0`.
The theorem is quantified over all `(tp, dp, ep)` satisfying the
2-way factorization — not just the specific (4, 2, 8) instance.

**"N independent replicas" scale-out**: everything above `world = tp × dp`
is orchestration, not distributed compute. Load-balance requests across
N (tp × dp)-sized replicas. Formal-verification-wise, the theorem verifies
ONE unit; N replicas is trivial composition.

### 4. Design cross-references

- **`TPGQA` reused unchanged** for attention. No weight_loader on TPGQA
  itself — it's a container module. `qkv_proj` and `o_proj` have their own
  weight_loaders. In real Qwen3, `q_norm` and `k_norm` (per-head RMSNorm)
  live at the attention level and WOULD need a facade weight_loader, but
  the bootcamp version skips them.
- **Gate is replicated within TP group.** Every TP peer runs identical
  routing on identical input (input is replicated within TP). No divergence,
  no split.
- **`self.experts`** is `nn.ModuleList` of `experts_per_rank` MLPs — sharded
  across EP.

### 5. Traps I hit (8 bugs in one review pass)

Wrote the whole HybridMoE.forward in one go, then reviewed against ref.
Every bug came out at the review, no runtime debugging needed:

**#1 — expert allocation size** (line 120 in first draft):
```python
self.experts = nn.ModuleList(RefSwiGLU_MLP(...) for _ in range(self.num_experts))
```
Should be `range(self.experts_per_rank)`. Test-scale invisible (weight_loader
fills only the correct low indices, compute never touches the rest), but at
Qwen3-30B-A3B this is an 8× memory blowup (1.2 GB extra per rank).

**Fix**: `range(self.experts_per_rank)`.

**#2 — token-of-record index formula, sort perm confusion**:
```python
top_k_expert_ids_perm_normalized = top_k_expert_ids_flat // self.top_k  # ← wrong tensor
x_repeated_k = x_flat[top_k_expert_ids_perm_normalized]                 # ← wrong base
```
Divided the *expert ids* by top_k (meaningless — those values are in
`[0, num_experts/top_k)`) instead of dividing the *sort perm* by top_k.
Then indexed `x_flat` (full striped input) instead of `local_x` (this
rank's stripe), so `tp_rank ≠ 0` grabbed rank 0's tokens.

**Fix**: `sorted_token_ids = perm // top_k; sorted_x = local_x[sorted_token_ids]`.

**Insight worth keeping**: `perm[i] // top_k` gives exactly the token id
for the record at sorted position `i`. Because in the unsorted flat layout,
positions `[j*top_k, (j+1)*top_k)` all belong to token `j`.

**#3 — dest-rank formula divisor**:
```python
dest_ranks = sorted_expert_ids // self.ep_size    # wrong
```
Should be `// self.experts_per_rank`. For num_experts=128, ep_size=8,
experts_per_rank=16: expert 15 lives on rank 0 (`15 // 16 = 0`), but
`// 8` gave `15 // 8 = 1` — routes expert 15 to rank 1. At test scale
(num_experts=8, ep_size=8, experts_per_rank=1), all `expert_id // 8 = 0`,
so EVERY record went to rank 0. Rank 0 flooded, ranks 1-7 idle.

**Fix**: `sorted_expert_ids // self.experts_per_rank`.

**#4 — bincount over global expert ids**:
```python
cnts = torch.bincount(expert_ids_local_T_sorted, minlength=self.experts_per_rank)
for idx, cnt in enumerate(cnts.tolist()):
    self.experts[idx](...)
```
Received expert ids are **global** (e.g., in `[48, 64)` for rank 3), but
`self.experts` is indexed by **local** slot in `[0, experts_per_rank)`.
`torch.bincount` returns a tensor of size `max(minlength, max_value+1)`,
so bincount over globals produces a size-`(max_global_id+1)` output; the
enumerate index equals the bin position (= global id), not a local slot.
Result: `self.experts[3]` on rank 3 is either a random-init expert (bug 1
present) or IndexError (bug 1 fixed).

**Root cause of misconception**: I assumed `torch.bincount` was
offset-based (counted relative to `min_value`). It's not — it counts
non-negative integers starting from 0. If you want offset counting, either
subtract the offset first (idiomatic) or use `torch.unique(..., return_counts=True)`.

**Fix**: `expert_ids_local_T -= self.expert_start` before the sort, so
values become local `[0, experts_per_rank)`.

**#5 — tuple unpacking on `.tolist()`**:
```python
for idx, cnt in (expert_ids_local_T_sorted_cnts.tolist()):
```
`.tolist()` on a 1D int tensor gives `[c0, c1, ...]`; iterating gives
`int`, not `(idx, cnt)` tuples. Raises `TypeError: cannot unpack non-iterable int`.

**Fix**: `for idx, cnt in enumerate(cnts.tolist()):`.

**#6 — un-sort direction**:
```python
y_repeated_k = y_repeated_k[top_k_expert_ids_perm]   # wrong direction
```
If `sorted, perm = torch.sort(orig)`, then `sorted[i] == orig[perm[i]]`.
Applying `y_sorted[perm]` again gives `orig[perm[perm[i]]]` — NOT recovery.

**Correct un-sort** (scatter, not gather):
```python
y_unsorted = torch.empty_like(y_sorted)
y_unsorted[perm] = y_sorted        # y_unsorted[perm[i]] = y_sorted[i]
```

This is the same pattern used correctly at Phase 8 (un-sort of received
records after local expert compute). Missed applying it symmetrically at
Phase 10 (un-sort of combined output before weight-multiply).

**Fix**: two buffers — `y_repeated_k_sorted` (post-combine, sorted) and
`y_repeated_k` (unsorted) — with `y_repeated_k[perm] = y_repeated_k_sorted`.

**#7 — CPU tensor into CUDA index**:
```python
local_y.index_add_(0, torch.arange(local_N).repeat_interleave(self.top_k), y_repeated_k)
```
`torch.arange(local_N)` without `device=` creates a CPU tensor; using it
as an index into a CUDA tensor errors.

**Fix**: `torch.arange(local_N, device=x.device)`.

**#8 — `norm_topk_prob=False` fancy indexing** (latent, test uses `True`):
```python
weights_top_k = F.softmax(router_logits)[top_k_expert_ids]
```
Indexing softmax output (shape `[local_N, num_experts]`) with `top_k_expert_ids`
(shape `[local_N, top_k]`) fancy-indexes on dim 0 → shape `[local_N, top_k, num_experts]`. Wrong.

**Fix**: `.gather(dim=-1, index=top_k_expert_ids)` — picks the k values at
positions specified by `top_k_expert_ids` along the last dim.

### 6. Meta-observations from the review pass

- **Index arithmetic bugs cluster**. #2, #3, #4, #6 are all "wrong divisor
  or wrong direction on a permutation." When implementing MoE from scratch,
  writing down each perm/id/dispatch/local mapping with concrete example
  numbers (like the `local_N=3, top_k=2, num_experts=4` trace I did later)
  catches these faster than staring at variable names.
- **Enumerate vs unique**. `torch.bincount` output positions are values,
  not sequential rank. If your bincount input is in a different index
  space than what you'll use to consume the output, you WILL trip. Subtract
  offsets first.
- **Naming discipline pays**. `top_k_expert_ids_perm_normalized` was a bad
  name — I picked "normalized" to signal my *intent* (a permutation index
  divided into token indices) but the actual value was `expert_ids // top_k`
  (no permutation involved). Better name: `sorted_token_ids` — describes
  what the values ARE, not what I wanted them to be.
- **Latent bugs are still bugs**. Bug #8's `norm_topk_prob=False` branch
  is never taken by the tests, but it's still broken code. Fix latent
  branches during review; running tests will not surface them.

### 7. Empirical benchmark (see [[project_ex08_topology_finding]])

Ex07 canonical vs Ex08 weiz-schedule benchmarks run on 8 GPUs (NVLink) and
16 GPUs (2 nodes across RoCE 25 GbE). Ex08 fused ends up roughly tied with
Ex07 canonical under **contiguous TP within nodes** (the natural
Qwen3-30B-A3B deployment), because the "avoid redundant intra-TP dispatch"
saving is intra-node bytes only. Ex08 would win under **straddled TP**
(TP > single-node GPU count), where intra-TP would otherwise ride
cross-node. Tabled the empirical validation of that regime.

### 8. Verification framing

The composition theorem for Ex07 canonical:

**Theorem (informal)**: For any `(tp_size, dp_size)` with `tp_size × dp_size == world_size`,
any well-formed partition of world into disjoint tp_groups (of size tp_size)
and disjoint dp_groups (of size dp_size), and `num_experts % ep_size == 0`
with `ep_size == world_size`, the HybridBlock forward pass produces output
equivalent to the single-GPU RefBlock forward (up to floating-point
associativity on the top-k combine sum), on input replicated within tp_group.

Broken into per-primitive lemmas:

1. **RMSNorm TP-invariance** (trivial): weight replicated → output replicated.
2. **TPGQA correctness** (Ex04 lemma): TP-sharded QKV + SDPA + RowParallel O
   preserves single-GPU semantics.
3. **HybridMoE correctness**: for each token `t` in `local_x`, and each
   of its top-k experts `e ∈ E_t`, the correctly-weighted output
   `experts[e](x_t) * w_{t,e}` is accumulated exactly once into `y_flat[t]`.
   Proof structure: track a single token+expert record through phases 0-11,
   show the record survives dispatch/combine + un-sort + all_gather to land
   at position `t` in `y_flat` on every TP peer.
4. **Residual composition**: `x + f(norm(x))` and `h + g(norm(h))` are
   TP-invariant when `f` and `g` are (from 1-3).

Proof-obligation cost: RMSNorm ~0 lines; TPGQA ~O(1) discharge from Ex04;
HybridMoE ~the meat (the phase-tracking argument); residual composition
~O(1). The formal verification track (Verus/Dafny/Z3) targets the HybridMoE
argument as its main proof obligation.

### Reference material worth revisiting

- [`bootcamp/ex07_tp_ep_hybrid/reference.py`](ex07_tp_ep_hybrid/reference.py)
  — the working canonical schedule; use as fresh-eyes read after solving.
- [`nanovllm-jun/nanovllm/models/qwen3.py::Qwen3SparseMoeBlock._forward_expert_parallel`](../nanovllm-jun/nanovllm/models/qwen3.py#L308)
  — production reference. Note the intern's version has TP and EP as
  mutually exclusive (`assert tp == 1 or ep == 1`); Ex07 is the composition
  they don't currently support.
- vLLM's [`FusedMoE`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe)
  — composes with vLLM's `ColumnParallelLinear` / `RowParallelLinear` on
  the attention side, giving exactly this TP+EP hybrid pattern in
  production.

