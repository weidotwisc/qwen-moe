# Verifying the nano-vLLM control plane in Rust/Verus — research note

Working note for the VeriCodeGen thread (Wei + Jun). Scopes the single-replica
scheduler as the first Verus target and writes down four candidate properties
plus a suggested attack. Not a paper draft.

## 1. Thesis

In an inference server the **control plane** — the scheduler and its KV-block
allocator — not the GPU kernels, is where the subtle correctness lives. Two
concrete data points from this project:

- **KV-cache OOB (`9dbe261`)** — `num_kvcache_blocks` was per-rank and
  unreconciled, so `store_kvcache` wrote out of bounds at `tp=1`. A control-plane
  *accounting-boundary* bug, not a kernel bug. Fixed by an `all_reduce` MIN.
- **DP lockstep invariant (`bootcamp/ex11`)** — a replica that skips its
  `ep_group` collective deadlocks the whole world; the measured "lockstep tax"
  runs 1.0×→~2.1× with imbalance/idle. A control-plane *scheduling* property.

Both argue for a verified control plane. This note takes the tractable half —
the **single-replica `Scheduler` + `BlockManager`** (`nanovllm/engine/scheduler.py`,
`block_manager.py`) — and pins down what "correct" means as candidate Verus specs.

## 2. System under verification

`Scheduler` = two deques (`waiting`, `running`) + a `BlockManager`, with
`schedule() / postprocess() / preempt() / is_finished()`. It is **pure Python
bookkeeping** — no tensors, no collectives, per-replica and completely local.
DP=N is just N independent `Scheduler`s wrapped by the `run_dp` lockstep;
`scheduler.py` is byte-identical across DP degrees. So the single-replica
scheduler is a **standalone, verifiable unit** (the EP=operator / DP=policy split);
the cross-replica lockstep is a separate, harder obligation — see §5.

## 3. The four properties (candidate specs)

### P1 — KV safety (memory safety of the block pool)
*The block pool is never over-committed, ownership is disjoint, every block is
freed exactly once, and no token is written outside its sequence's blocks.*

Invariants over `BlockManager` state:
- **Conservation**: `Σ blocks_held(seq) + free_blocks == num_kvcache_blocks`.
- **Disjointness**: no physical block appears in two sequences' block tables.
- `schedule` prefill: `requires can_allocate(seq)`; `ensures` it takes exactly
  `ceil(num_tokens/block_size) − cached` blocks from the free pool.
- decode append: `requires can_append(seq)`; `ensures` the written slot lies
  inside `seq`'s last allocated block (no OOB).
- `postprocess`(FINISHED) / `preempt`: `ensures deallocate` returns the blocks
  and drops ownership (no leak, no double-free).

Failure mode = **`9dbe261`**: capacity in the conservation invariant was the
per-rank count, not the global min → `can_allocate` over-optimistic → OOB write.
Code: `scheduler.py:36, 44-46, 60, 69, 91`; `block_manager.{can_allocate,allocate,can_append,may_append,deallocate}`.
Verus fit: **strong** (bounds + no-overflow + the conservation/disjointness invariants).

### P2 — Termination correctness
*A sequence becomes FINISHED exactly on EOS-or-max_tokens, and a chunked prefill
emits no completion token until the prompt is fully prefilled.*

`postprocess` `ensures`:
- `seq.FINISHED ⟺ (token == eos ∧ ¬ignore_eos) ∨ num_completion_tokens == max_tokens`
  (evaluated after `append_token`);
- `is_prefill ∧ num_cached_tokens < num_tokens ⇒` no `append_token` this step
  (the `continue` guard, `scheduler.py:86`);
- `num_completion_tokens ≤ max_tokens` always (never overshoot).

Failure mode: off-by-one length, or appending mid-chunked-prefill (double-count /
corrupt output). Code: `scheduler.py:86-92`. Verus fit: **strong** (functional
postcondition).

### P3 — Liveness / no-livelock (progress)
*Ideal: every admitted sequence eventually reaches FINISHED; preemption does not
livelock and prefill-first does not starve decode.*

Verus is for safety + functional correctness, **not** temporal/fairness liveness.
Verify **safety surrogates** instead, and defer true fairness:
- **No lost sequence**: every admitted seq is in exactly one of
  `{waiting, running, finished}` at all times; `preempt` moves running→`waiting`
  *front* (`appendleft`, `scheduler.py:79`), never drops it.
- **Loop termination** (`decreases`): `schedule`'s prefill loop decreases
  `|waiting|` or the token budget; the decode loop pops `running`; each `preempt`
  pops one seq from `running`, so `|running|` strictly decreases → the inner
  block-pressure loop terminates within one `schedule` call.
- **Preemption frees ≥1 block**, so a preempted step makes room its successor can
  use (no busy-spin at fixed state).

True fairness/starvation-freedom → TLA+ / model-checking (temporal logic), or an
informal argument; keep out of the Verus core. Verus fit: **partial** (surrogates
yes, temporal liveness no).

### P4 — Batch budget
*Every scheduled batch respects the seq and token caps; a decode batch is one
token per sequence.*

`schedule` `ensures`:
- `|scheduled_seqs| ≤ max_num_seqs`;
- `is_prefill ⇒ Σ num_scheduled_tokens ≤ max_num_batched_tokens`;
- `¬is_prefill ⇒ ∀ seq. num_scheduled_tokens == 1`.

Failure mode: blow the token budget → OOM or exceed a kernel/graph bound (the
`run_model` >512 eager branch, the captured CUDA-graph sizes). Code:
`scheduler.py:30-33, 42, 46-47, 67`. Verus fit: **strong** (simple postcondition).

## 4. Suggestion — how to attack it

- **Remodel, don't translate.** Verus verifies Rust, so hand-port the
  `Scheduler` + `BlockManager` to a Rust state machine mirroring the two deques
  and a block free-list/bitmap. Keep it a faithful spec-level model (the
  brown-field artifact framing), AI-draft the Verus proofs, Jun audits — the
  workflow from `[[project-jun-audited-proofs]]`.
- **Order by fit.** P1, P2, P4 first (functional + invariants + `decreases` — the
  core Verus sweet spot). P3 as safety surrogates only. This front-loads the
  provable, high-value KV-safety result.
- **Encode the boundary assumption.** `9dbe261` lived at the scheduler↔mesh
  boundary. Make `num_kvcache_blocks` a **spec precondition** ("= reconciled
  global-min across ranks"); the module is verified *relative to* that, and the
  reconciliation (`all_reduce` MIN) is a separate obligation. The lesson: a
  verified scheduler that silently assumed a per-rank count would *not* have
  caught the bug — the value is in writing the assumption down and discharging it
  where it actually holds.
- **Milestones.** (a) Rust `BlockManager` + P1; (b) `schedule/postprocess` + P2/P4;
  (c) `preempt` + P3 surrogates; (d) the block-count precondition + a note on the
  reconciliation obligation; (e) write up as the paper's motivating verified
  artifact, with the OOB bug as the "what verification would have caught" hook.

## 5. Out of initial scope — the DP lockstep property (P5, concurrency)

The `ex11` invariant — *every rank posts an identical collective schedule per
step; idle → dummy forward* — is distributed **deadlock-freedom / liveness**, not
single-replica safety. Different verification character (concurrency, collective
schedules) → TLA+ or session types, not Verus. Keep it as a companion result: the
deadlock demo and the lockstep-tax numbers already exist in `bootcamp/ex11`.

## 6. Integration architecture — run the verified code, don't just model it

Verus verifies *ordinary* Rust: the proof annotations (`requires`/`ensures`/
`invariant`/`decreases`, `spec`/`proof`/ghost code) are erased at compile time and
the `exec` functions compile with cargo like any crate. So the verified library
**is** the runnable one — we don't have to choose between "proven" and "in the
engine." §2-§4 describe it as a model; this section makes it the real scheduler.

**Chosen shape: Python drives, calls into a Rust `sched_block_lib`** (via PyO3 +
maturin). Keep nanovllm's Python for everything hard — process spawn,
`model_runner` (forward/sampling/CUDA graphs), NCCL, shm IPC, the `run_dp`
lockstep, tokenizer. Extract **only `scheduler` + `block_manager`** into the Rust
crate; `LLMEngine` calls it where it calls `self.scheduler.*` today. This is the
`tokenizers` / `polars` / `pydantic-core` pattern (Rust core, PyO3 boundary,
Python driver). Rejected the inverse (Rust drives, embeds CPython): it would
reimplement spawn + shm + NCCL orchestration in FFI for zero verification benefit
(orchestration is P5, not the Verus target). Don't invert control.

**Why it's tractable: the boundary is small and data-only — no tensors cross.**

```
Python (driver)                       Rust sched_block_lib (Verus-verified core)
  add_request(meta) ─────────────────► push to waiting  (seq_id, prompt_len, max_tokens, eos)
  schedule() ◄───────────────────────► (seq_ids, is_prefill,
                                         per-seq: block_table[int], num_scheduled, num_cached)
    → model_runner builds CUDA tensors from those block ids   (Python side)
    → forward + sample                                        (Python / CUDA / NCCL)
  postprocess(seq_ids, token_ids) ───► append / mark finished / free blocks
  is_finished() ◄────────────────────  bool
```

Rust owns the **control + allocation state** (the two queues, the block free-list,
per-seq counts). Python keeps the **token buffers** and builds tensors from the
block ids Rust returns. Only ints/flags/small structs cross the FFI — no GPU
handles, no tensors.

**TCB — "verified core + trusted boundary."** Verus proves the pure Rust
functions (`schedule`, `allocate`, `postprocess`, …). The PyO3 shim that
deserializes Python args → Rust structs and back is **not** verified and *is* what
must establish the preconditions (valid, in-range inputs). Paper claim must read:
*verified scheduler modulo a thin trusted FFI shim* — the TCB is the shim + the
SMT solver, not the whole path.

**Three caveats to scope up front:**
1. **P1's block-count precondition still comes from Python.** The `all_reduce` MIN
   reconciliation (`9dbe261`) stays in `model_runner`; `sched_block_lib` takes the
   reconciled global-min as a `requires` and does not prove the collective — it
   consumes its result (the obligation split from §4).
2. **Prefix-cache hashing reads token ids** (`block_manager.hash_blocks`) — the one
   thing that would pull token payloads across the boundary. v1: pass precomputed
   hashes in, or drop prefix caching in the verified lib and add later. Scope out.
3. **Verus `exec` subset** — deques/vecs/ints/sets are fine (the scheduler fits);
   just avoid arbitrary std in the verified core, keep it in the shim.

**Bonus — a real perf axis, not just verification.** The scheduler sits on rank
0's critical path (the single-controller stall vLLM V1 fixed with async
scheduling). A Rust scheduler is faster and can **release the GIL**
(`Python::allow_threads`) during the call, trimming per-step scheduler overhead off
the GPU's critical path. So the artifact is "verified control plane that also
shaves the rank-0 stall" — a second axis for the paper beyond correctness.

**Milestone (f):** after §4's (a)-(e), wrap the verified crate with a PyO3 shim,
swap `LLMEngine` to call `sched_block_lib`, and re-run the GSM8K + throughput
harness to confirm parity (accuracy identical, per-step overhead ≤ Python).

## Links
`[[project-data-control-plane-split]]` · KV bug `9dbe261` · `bootcamp/ex11` ·
`[[project-jun-audited-proofs]]` · `[[project-paper-reframe]]`.
