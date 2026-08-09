# Bootcamp learning summary

Personal notes on what I picked up in each exercise. Grows as I finish more.

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

### Reference material worth revisiting

- [bootcamp/ex00_dist_primer/README.md](ex00_dist_primer/README.md) — MPI ↔ torch.dist mapping table.
- [bootcamp/ex00_dist_primer/solution.py](ex00_dist_primer/solution.py) — the specs I wrote.
- nanovllm-jun's `_forward_expert_parallel` — real-world EP dispatch using these primitives.

---

## Ex01 — Linear Tensor Parallelism

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

### 5. Deferred: the pattern overhaul (revisit after Ex02)

My Ex01 currently:
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

<!-- Add Ex02 summary below when done. -->
