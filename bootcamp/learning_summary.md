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
| Ex05a | MoE baseline (naive per-expert loop) | ⬜ pending | – | Batch 2 (Claude scaffolds next) |
| Ex05b | MoE permuted (grouped compute) | ⬜ pending | – | Batch 2 |
| Ex06 | Expert Parallelism (all-to-all) | ⬜ pending | – | Design chat with Claude first |
| Ex07 | TP + EP hybrid | ⬜ pending | – | Composition theorem — paper's main artifact |
| Ex08 | Fused MoE Triton kernel | ⬜ pending | – | Port Ex05b to Triton |

**Bugs-caught count so far**: 13 distinct bugs across Ex01 + Ex02 + Ex03 + Ex04 — see the "Traps I hit" sections in each entry for the failure mode and fix.

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

<!-- Add Ex05 summary below when done. -->
