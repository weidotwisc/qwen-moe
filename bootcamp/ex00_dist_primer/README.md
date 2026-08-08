# Exercise 0 — torch.distributed primer

Warm-up. **Seven tasks**: two init/teardown wrappers (`init_dist`,
`destroy_dist`) plus five collective wrappers. Written with
formal-verification in mind — each has a docstring pre/post condition that
will be treated as an opaque spec by the paper's proof.

## The init sequence (Tasks 1-2)

Every torch.distributed script has to bring up a process group before the
first collective and tear it down at the end. In your test harness this is
done by `bootcamp/dist_utils.py::_init`, but you should be able to write it
yourself — it's what every real training script (torchrun-launched,
Accelerate, DeepSpeed, FSDP) does at the top of its worker.

The canonical form, annotated:

```python
def init_dist(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"        # (a) rendezvous host
    os.environ["MASTER_PORT"] = str(port)          # (b) rendezvous port
    os.environ.setdefault(
        "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")    # (c) crash-on-fail
    torch.cuda.set_device(rank)                    # (d) MUST precede init
    dist.init_process_group(                       # (e) rendezvous + backend
        "nccl", rank=rank, world_size=world_size)

def destroy_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
```

**Why each line:**

- **(a, b) MASTER_ADDR / MASTER_PORT** — the rendezvous endpoint. Every rank
  reads these env vars in `init_process_group` to find the coordinator. With
  torchrun, the launcher sets these for you; with `mp.spawn` you set them
  yourself. Port has to be free on `MASTER_ADDR`; we rotate the port
  per-`run_on_ranks` call because back-to-back tests hit TIME_WAIT on the
  same port.
- **(c) TORCH_NCCL_ASYNC_ERROR_HANDLING=1** — makes NCCL raise a Python
  exception on the offending rank instead of hanging when a collective
  fails mid-run. Was called `NCCL_ASYNC_ERROR_HANDLING` before torch 2.2;
  the old name still works but emits a deprecation warning.
- **(d) `torch.cuda.set_device(rank)` before `init_process_group`** — this
  is the ordering trap. NCCL binds each rank to whichever CUDA device is
  current at init time. If two ranks are current on the same device
  (because you forgot `set_device`, or set it after init), NCCL will
  **silently deadlock or corrupt buffers** at the first collective — not
  raise. This is a real gotcha, present in almost every torch.dist bug
  report I've seen. Verifiable via the post-condition
  `torch.cuda.current_device() == rank`.
- **(e) `init_process_group("nccl", ...)`** — the actual rendezvous.
  Backend "nccl" for GPU; "gloo" for CPU (rarely useful except in tests).
  Blocks until all `world_size` ranks have called this. If one rank never
  calls it, everyone hangs — the classic distributed-init deadlock.

**Idempotence of `destroy_dist`** — the `if dist.is_initialized()` guard
matters for the test harness's `finally:` block: if `init_dist` itself
failed (bad port, race), we still want `destroy_dist` to be a no-op rather
than raise a second exception that masks the first.

## MPI → torch.dist mapping

| MPI | torch.dist | Notes |
|---|---|---|
| `MPI_Allreduce(SUM)` | `dist.all_reduce(x, op=SUM)` | In-place. Returns None (in torch), not `MPI_SUCCESS`. |
| `MPI_Allgather` | `dist.all_gather_into_tensor(out, x)` | Preferred over `dist.all_gather(list, x)` which allocates a list of tensors. |
| `MPI_Reduce_scatter_block(SUM)` | `dist.reduce_scatter_tensor(out, x, op=SUM)` | Input `x` must have leading dim == world_size. |
| `MPI_Alltoall` | `dist.all_to_all_single(out, x)` — equal splits case | When output/input split sizes are omitted, equal splits assumed. |
| `MPI_Alltoallv` | `dist.all_to_all_single(out, x, output_split_sizes=os, input_split_sizes=is)` | **The EP dispatch primitive.** |
| `MPI_Isend/Irecv + MPI_Wait` | `req = dist.all_reduce(x, async_op=True); req.wait()` | Returns a `Work` handle (torch's equivalent of `MPI_Request`). |
| `MPI_Comm_split` / `MPI_Group` | `dist.new_group(ranks=[0,1,2,3])` | Sub-communicators for TP/EP separation. |

## Non-obvious differences from MPI (real gotchas)

**In-place semantics.** `dist.all_reduce(x)` mutates `x` and returns `None`.
Not the return value. Do not write `x = dist.all_reduce(x)` — you'll get `None`.

**NCCL device binding.** Every rank must call `torch.cuda.set_device(rank)`
before `dist.init_process_group("nccl", ...)`. If two ranks bind the same
device, NCCL will silently deadlock. `bootcamp/dist_utils.py::_init` handles
this — you don't have to.

**Tensor contiguity.** All collectives require `.is_contiguous()`. A `narrow`
result is a view; call `.contiguous()` before passing to a collective.

**Dtype consistency across ranks.** NCCL doesn't check that all ranks agree
on the dtype/shape of a collective's input. A mismatch = corruption or
deadlock, not an error. Formally, the paper's precondition on every wrapper
below requires *matching* shape and dtype across `group`.

**Groups are handles, not ranks lists.** `dist.new_group([0, 1, 2, 3])`
returns a `ProcessGroup` handle. You pass that handle to `group=`. Members
of the group refer to themselves by their *global* rank when calling into
the group's collectives — torch translates internally.

**Non-member ranks pass `None`.** If a rank is not in a sub-group, it should
still execute the outer control flow but pass `group=None` OR just skip
the call. **Never mix**: some ranks calling `foo(group=g)` while others call
`foo(group=None)` on the same collective is a deadlock. In this course we
avoid this by keeping every rank a member of every group it might touch.

**Async work handles.** `req = dist.all_reduce(x, async_op=True)` returns a
`Work` object with `.wait()` and `.is_completed()`. Equivalent to
`MPI_Request` + `MPI_Wait`. Useful for comm/compute overlap in ex07.
`req.wait()` blocks until the collective completes (does NOT poll).

## What to fill in

Five one-line wrappers in [solution.py](solution.py). Each is 2-4 lines of
Python once you know the API. The value is in reading the docstring and
matching its pre/post condition exactly — those specs become part of the
proof obligations.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest -x bootcamp/tests/test_ex00_dist_primer.py -v
```

Green pytest = wrappers correct.  Move on to ex01.

## For the paper's verification story

The wrappers are the atomic units the paper models. Every module in ex06-08
uses them exclusively — never a raw `dist.*` call. This means:

- The abstract state machine for EP dispatch has exactly 4 events:
  `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all` (equal + variable).
- Deadlock freedom = "every rank calls the same sequence of these atoms".
- Data-race freedom = "no shared mutable state outside the wrappers".
- Functional correctness = "post-condition of each wrapper implies the
  intended computation".

Keep the wrapper implementations tight; the docstring specs are load-bearing.
