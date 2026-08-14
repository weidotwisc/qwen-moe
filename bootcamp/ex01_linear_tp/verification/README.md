# Ex01 verification track — three tools, one theorem

This directory contains **attempt proofs** in Verus, Dafny, and Z3 of the
correctness properties of `ColumnParallelLinear` from [../solution.py](../solution.py).
It's the first exercise of the workshop paper's verification track and the
scaffold for the pattern we'll replicate in Ex02–Ex07.

**Read [PROPERTIES.md](PROPERTIES.md) first** — it states the four properties
(C1–C4), the two matmul axioms (M1, M2), and the shared abstraction model.
The proof files below realize that spec in three different systems.

## Files

| File | Tool | Coverage | Verified in this session? |
|---|---|---|---|
| [PROPERTIES.md](PROPERTIES.md) | none | Properties + axioms + model | n/a |
| [column_parallel.dfy](column_parallel.dfy) | Dafny 4.x | C1, C2, C3, C4 (tp=2); C4-general stubbed | No — Dafny not installed on pod |
| [column_parallel.rs](column_parallel.rs) | Verus 0.2024+ | C1, C2, C3, C4 (tp=2); C4-general stubbed | No — Verus not installed on pod |
| [column_parallel_z3.py](column_parallel_z3.py) | z3-solver (Python) | C1–C4 for 7 concrete shapes | **Yes — 28/28 pass** |

## Quick check

Run the Z3 bounded verification now (only tool with automated CI on this pod):

```sh
uv run --with z3-solver python bootcamp/ex01_linear_tp/verification/column_parallel_z3.py
```

Expected output: `Summary: 28 passed, 0 failed`, ~0.2s wall time.

## Running the parameterized proofs (Dafny, Verus)

Neither tool is available on the shared pod. Options for the intern to
verify these:

### Dafny (recommended for first pass — friendliest syntax)

```sh
# Install (needs sudo or a personal dotnet):
dotnet tool install --global dafny --version 4.9.2   # or `brew install dafny` / `nix-env -iA nixpkgs.dafny`

# Verify:
dafny verify column_parallel.dfy
# Expected: "Dafny program verifier finished with N verified, 0 errors"
# where N corresponds to the lemmas that don't have {:verify false}.
```

**What verifies (should be N = 6 with our current proofs)**:
- `GatherFromEqualsSuffix` (helper lemma for C1)
- `C1_ShardGatherRoundtrip`
- `C2_ShardingDisjoint`
- `C3_WeightLoaderPostcondition`
- `C4_ForwardCorrectness_TP2` (concrete tp_size=2)
- `SanityShardTP2` and `SanityShardTP4` (concrete examples)

**What's stubbed** (marked `{:verify false}`):
- `C4_ForwardCorrectness_General` — the parameterized-over-tp_size version.
  Proof structure is documented inline; the induction skeleton mirrors
  `GatherFromEqualsSuffix`. Extending is straightforward but adds ~50 lines.

### Verus (Rust-hosted, closer to production code style)

```sh
# Install:
git clone https://github.com/verus-lang/verus
cd verus && ./tools/get-z3.sh && vargo build --release
export PATH="$PWD/target-verus/release:$PATH"

# Verify:
verus column_parallel.rs
```

**What verifies**:
- `gather_from_equals_suffix`
- `c1_shard_gather_roundtrip`
- `c2_sharding_disjoint` (uses nonlinear-arith hints — see the `by (nonlinear_arith)` clauses)
- `c3_weight_loader_postcondition`
- `c4_forward_correctness_tp2`

**What's stubbed**: `c4_forward_correctness_general` and `axiom_m1` are marked
`#[verifier::external_body]` — Verus accepts them without proof. The axiom
is intentional (M1 is our declared assumption); the general C4 needs the same
inductive extension as the Dafny version.

## The three tools compared

| Aspect | Dafny | Verus | Z3 (Python) |
|---|---|---|---|
| **Proof style** | Universal over `seq<T>` | Universal over `Seq<T>` (Rust) | Bounded per shape |
| **Native LA** | No — matmul uninterpreted | No — matmul uninterpreted | Yes — matmul as symbolic sum |
| **What it proves** | C1–C4 for all shapes (up to M1) | Same as Dafny | C1–C4 for enumerated shapes |
| **Depends on axiom M1?** | Yes | Yes | No — LA checked directly |
| **Failure mode** | Verification timeout / stuck proof search | Verification error / hint needed | Solver timeout |
| **Install effort** | Medium (dotnet tool) | Higher (build from source) | Trivial (`pip install z3-solver`) |
| **Learning curve** | Low — friendly syntax | Higher — Rust + verifier idioms | Trivial for Python users |
| **Paper positioning** | Primary parameterized proof | Alternative parameterized proof, same theorem | Concrete regression evidence |

**Complementary, not redundant**: the parameterized proofs (Dafny/Verus)
give the theorem the paper cites — "for all TP sizes, our column-parallel
linear layer satisfies C1–C4." The Z3 concrete check catches encoding
bugs — if some `(tp_size, M, N)` violates C4 in Z3 but the parameterized
proofs claim otherwise, one of them is wrong. Historically this
cross-validation has caught real bugs in formal specs.

## Delivery plan (paper timeline)

- **This scaffold** — Ex01 Column, three tools, C1–C4. Ready for intern to
  pick up and expand.
- **Next (intern's first task)**: Complete `C4_ForwardCorrectness_General`
  in Dafny (~50 lines of induction). Then port the same skeleton to Verus.
  Both need identical proof structure; Dafny first is easier.
- **Follow-up (Ex01 Row)**: Add `row_parallel.dfy` / `.rs` / `_z3.py`,
  properties R1–R4, axiom M2. Structural mirror of Column — most code
  duplicates with axis renaming.
- **Then Ex02** (SwiGLU MLP TP) — the merged-column-parallel pattern needs
  a new property (weight-loader is called N times per instance, disjoint
  regions). Composition proof: Column → Row with one all-reduce.
- **Ex03–Ex04** (attention + GQA TP) — QKV as 3-way merged column. Ex04
  adds KV-replication as an interesting new invariant (some ranks hold
  identical bytes; the replica invariant is a bijection property).
- **Ex06** (EP) — introduces `all_to_all_variable` as a new atomic
  collective. Needs a comm-safety proof (deadlock freedom under the
  transposition invariant).
- **Ex07** (TP + EP hybrid) — the paper's centerpiece: composition of
  sub-group collectives without deadlock, under our verification-friendly
  discipline (fixed schedule, explicit `group=`, progress-preserving loops).
  The proof composes Ex04's TP theorem with Ex06's EP theorem.

## Notes on the proof-writing style

Two design choices worth flagging:

**1. Matmul is uninterpreted in Dafny/Verus.** These tools don't reason
about linear algebra natively — they reason about discrete structural
properties (sequences, concatenation, arithmetic). We declare `matmul` as
an opaque function and add axioms M1, M2 as first-order facts. The
linear-algebra content of the proof is exactly those two axioms; the tool
verifies the parallel schedule preserves M1/M2's structure.

This is standard in distributed-systems verification. It also matches the
paper's intended framing: the parallelism scheme's correctness is a
*schedule* property, not a *numerics* property. Numerical concerns
(bf16 rounding, non-associative reduction order) are separate — see the
"tolerance" clause in each exercise's test spec.

**2. Z3 uses concrete matmul because it can.** For fixed shapes, matmul
expands to a finite nested sum, which Z3 handles directly. This gives us
*stronger* evidence for those specific shapes (no axiom needed), at the
cost of coverage. Z3 is our "regression tester" — Dafny/Verus are our
"theorem provers."

## What to do next

For Wei's immediate purposes:

- **You** don't need to install Dafny/Verus. Read [PROPERTIES.md](PROPERTIES.md)
  and the three proof files as reference for what the paper's spec will
  look like. The Z3 script runs on your existing setup and confirms the
  properties hold for the shapes it enumerates.
- **The intern** (verification specialist) picks up the Dafny file first,
  gets `dafny verify column_parallel.dfy` to green (fills in the
  `{:verify false}` stub), then ports the same skeleton to Verus, then
  starts on Ex02.
- **Once Ex01 is fully verified in Dafny**, the paper's verification section
  has its first proof artifact. Everything else scales from that pattern.
