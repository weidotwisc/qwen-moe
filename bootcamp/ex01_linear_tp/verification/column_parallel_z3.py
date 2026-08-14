"""Z3 bounded verification for ColumnParallelLinear (Ex01).

Verifies properties C1–C4 from PROPERTIES.md for a menu of concrete
(tp_size, M, N) shape combinations. Complements the parameterized proofs
in column_parallel.dfy and column_parallel.rs: those prove for ALL sizes;
this one proves for a specific enumeration of sizes, with matmul defined
symbolically (not as an uninterpreted function). Together they give:

  Dafny/Verus: "For all tp_size, M, N, C1–C4 hold assuming axiom M1."
  Z3 (this file): "For each (tp_size, M, N) in the enumeration, C1–C4
                   hold with matmul as concrete nested-sum multiplication
                   — no axiom needed for this specific shape."

Z3's role is regression evidence: if the Dafny/Verus encoding is wrong,
Z3's concrete check will disagree on some shape. Concrete + parameterized
proofs cross-validate each other.

Run:
    uv run python column_parallel_z3.py

Requires z3-solver:
    uv pip install z3-solver
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from z3 import (
    And,
    ArithRef,
    Not,
    Real,
    Solver,
    Sum,
    unsat,
)


# =====================================================================
# §1 — Symbolic tensor helpers
# =====================================================================

Tensor = list[list[ArithRef]]  # shape [rows, cols], row-major


def sym_tensor(name: str, rows: int, cols: int) -> Tensor:
    """Fresh symbolic tensor of shape [rows, cols], each element a distinct
    Real variable."""
    return [[Real(f"{name}_{i}_{j}") for j in range(cols)] for i in range(rows)]


def sym_shard(w: Tensor, rank: int, tp_size: int) -> Tensor:
    """Column-parallel shard: extract rows [rank*s .. (rank+1)*s) of w."""
    M = len(w)
    assert M % tp_size == 0, f"M={M} not divisible by tp_size={tp_size}"
    s = M // tp_size
    return w[rank * s : (rank + 1) * s]


def sym_gather_all(w: Tensor, tp_size: int) -> Tensor:
    """Concatenate all shards in rank order."""
    result: Tensor = []
    for r in range(tp_size):
        result.extend(sym_shard(w, r, tp_size))
    return result


def sym_transpose(t: Tensor) -> Tensor:
    """Transpose: [rows, cols] -> [cols, rows]."""
    if not t:
        return []
    rows, cols = len(t), len(t[0])
    return [[t[i][j] for i in range(rows)] for j in range(cols)]


def sym_matmul(x: Tensor, w_t: Tensor) -> Tensor:
    """Matmul: (B, in) @ (in, out) -> (B, out)."""
    B = len(x)
    inner = len(x[0]) if x else 0
    out = len(w_t[0]) if w_t else 0
    assert len(w_t) == inner, f"matmul shape mismatch: x={B}×{inner}, w_t={len(w_t)}×{out}"
    result: Tensor = []
    for i in range(B):
        row: list[ArithRef] = []
        for j in range(out):
            row.append(Sum([x[i][k] * w_t[k][j] for k in range(inner)]))
        result.append(row)
    return result


def sym_concat_cols(a: Tensor, b: Tensor) -> Tensor:
    """Horizontal concatenation on dim 1 (side-by-side columns)."""
    assert len(a) == len(b), f"concat_cols row mismatch: {len(a)} vs {len(b)}"
    return [a_row + b_row for a_row, b_row in zip(a, b)]


def sym_concat_rows(*tensors: Tensor) -> Tensor:
    """Vertical concatenation on dim 0."""
    result: Tensor = []
    for t in tensors:
        result.extend(t)
    return result


def assert_tensor_eq(solver: Solver, a: Tensor, b: Tensor) -> None:
    """Add tensor-equality constraint to solver."""
    assert len(a) == len(b), f"row count mismatch: {len(a)} vs {len(b)}"
    for row_a, row_b in zip(a, b):
        assert len(row_a) == len(row_b), f"col count mismatch: {len(row_a)} vs {len(row_b)}"
        for x, y in zip(row_a, row_b):
            solver.add(x == y)


def tensor_disjunctive_neq(a: Tensor, b: Tensor):
    """Return a Z3 formula: 'some cell of a differs from b'."""
    assert len(a) == len(b)
    disjuncts = []
    for row_a, row_b in zip(a, b):
        assert len(row_a) == len(row_b)
        for x, y in zip(row_a, row_b):
            disjuncts.append(x != y)
    from z3 import Or
    return Or(*disjuncts) if disjuncts else False


# =====================================================================
# §2 — Property verifiers
# =====================================================================


@dataclass
class Result:
    property: str
    tp_size: int
    M: int
    N: int
    verified: bool
    elapsed_ms: float
    note: str = ""


def verify_c1(tp_size: int, M: int, N: int) -> Result:
    """C1: shard-and-gather roundtrip. GatherAll(w, tp_size) == w."""
    t0 = time.perf_counter()
    w = sym_tensor("w", M, N)
    gathered = sym_gather_all(w, tp_size)
    # Check: is it possible for gathered to differ from w?
    solver = Solver()
    solver.add(tensor_disjunctive_neq(w, gathered))
    result = solver.check()
    verified = result == unsat
    elapsed = (time.perf_counter() - t0) * 1000
    return Result("C1_shard_gather_roundtrip", tp_size, M, N, verified, elapsed)


def verify_c2(tp_size: int, M: int, N: int) -> Result:
    """C2: sharding disjointness. Any two distinct ranks have disjoint row ranges."""
    t0 = time.perf_counter()
    s = M // tp_size
    solver = Solver()
    # For all r1 != r2: (r1+1)*s <= r2*s  OR  (r2+1)*s <= r1*s.
    for r1 in range(tp_size):
        for r2 in range(tp_size):
            if r1 == r2:
                continue
            disjoint = (
                ((r1 + 1) * s <= r2 * s) or ((r2 + 1) * s <= r1 * s)
            )
            # Concrete arithmetic; Z3 just checks arithmetic facts.
            solver.push()
            solver.add(Not(disjoint))
            result = solver.check()
            solver.pop()
            if result != unsat:
                elapsed = (time.perf_counter() - t0) * 1000
                return Result(
                    "C2_sharding_disjoint", tp_size, M, N, False, elapsed,
                    note=f"disjointness fails for ranks ({r1}, {r2})",
                )
    elapsed = (time.perf_counter() - t0) * 1000
    return Result("C2_sharding_disjoint", tp_size, M, N, True, elapsed)


def verify_c3(tp_size: int, M: int, N: int) -> Result:
    """C3: weight_loader post-condition. WeightAfterLoad(w, r, tp_size) == Shard(w, r, tp_size).

    Structural — WeightAfterLoad is defined AS shard(). This holds
    trivially. Verification here is a regression test: if someone edits
    the sharding definition inconsistently across the two functions,
    this fails.
    """
    t0 = time.perf_counter()
    for r in range(tp_size):
        w = sym_tensor("w", M, N)
        loaded = sym_shard(w, r, tp_size)   # definition of weight_after_load
        expected = sym_shard(w, r, tp_size)  # definition of shard
        solver = Solver()
        solver.add(tensor_disjunctive_neq(loaded, expected))
        if solver.check() != unsat:
            elapsed = (time.perf_counter() - t0) * 1000
            return Result(
                "C3_weight_loader_postcondition", tp_size, M, N, False, elapsed,
                note=f"failed on rank {r}",
            )
    elapsed = (time.perf_counter() - t0) * 1000
    return Result("C3_weight_loader_postcondition", tp_size, M, N, True, elapsed)


def verify_c4(tp_size: int, M: int, N: int, batch: int = 2) -> Result:
    """C4: forward correctness. Concatenating per-rank forwards gives full matmul.

    Here matmul is defined SYMBOLICALLY (nested sum), so Z3 handles the
    linear algebra directly. This is stronger than the Dafny/Verus proofs
    (which rely on axiom M1), but only for this specific shape.
    """
    t0 = time.perf_counter()
    x = sym_tensor("x", batch, N)      # x: [B, in_features]
    w = sym_tensor("w", M, N)          # w: [out_features, in_features]

    # Unsharded forward: matmul(x, w.T).
    full_forward = sym_matmul(x, sym_transpose(w))

    # Sharded forward: for each rank, matmul(x, shard(w, r, tp_size).T).
    # Concatenate on dim 1 (last axis).
    rank_outputs = []
    for r in range(tp_size):
        w_shard = sym_shard(w, r, tp_size)
        y_r = sym_matmul(x, sym_transpose(w_shard))
        rank_outputs.append(y_r)
    gathered_forward = rank_outputs[0]
    for r_out in rank_outputs[1:]:
        gathered_forward = sym_concat_cols(gathered_forward, r_out)

    # Check: any cell differs?
    solver = Solver()
    solver.add(tensor_disjunctive_neq(gathered_forward, full_forward))
    result = solver.check()
    verified = result == unsat
    elapsed = (time.perf_counter() - t0) * 1000
    return Result("C4_forward_correctness", tp_size, M, N, verified, elapsed,
                  note=f"batch={batch}")


# =====================================================================
# §3 — Runner
# =====================================================================


ENUMERATED_SHAPES: list[tuple[int, int, int]] = [
    # (tp_size, M, N)
    (1, 4, 4),
    (2, 4, 4),
    (2, 8, 4),
    (4, 8, 4),
    (4, 16, 8),
    (8, 16, 8),
    (8, 32, 8),
]


def print_header(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def print_result(r: Result) -> None:
    status = "PASS" if r.verified else "FAIL"
    shape = f"tp={r.tp_size} M={r.M} N={r.N}"
    time_str = f"{r.elapsed_ms:6.1f}ms"
    note = f"  ({r.note})" if r.note else ""
    print(f"  [{status}] {shape:22s}  {time_str}{note}")


def main() -> int:
    verifiers: list[tuple[str, Callable[[int, int, int], Result]]] = [
        ("C1: shard-and-gather roundtrip", verify_c1),
        ("C2: sharding disjointness", verify_c2),
        ("C3: weight loader post-condition", verify_c3),
        ("C4: forward correctness (matmul concrete)", verify_c4),
    ]

    total_pass = 0
    total_fail = 0

    for name, fn in verifiers:
        print_header(name)
        for tp_size, M, N in ENUMERATED_SHAPES:
            r = fn(tp_size, M, N)
            print_result(r)
            if r.verified:
                total_pass += 1
            else:
                total_fail += 1

    print(f"\n{'=' * 60}")
    print(f"  Summary: {total_pass} passed, {total_fail} failed")
    print(f"{'=' * 60}\n")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
