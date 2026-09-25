"""Tests for Ex11 — the DP lockstep invariant + the lockstep tax.

Unlike ex01-ex08 (numerical equivalence vs a single-GPU oracle), ex11's core is a
CONTROL-PLANE property: with `ep_group == world`, every DP replica must co-arrive
at every layer, so an idle replica must still post its collective (the dummy
forward) or the whole world deadlocks. The tests therefore check:

  test_lockstep_tax*        the analytical tax formula (CPU, no GPU).
  test_lockstep_completes   with the dummy forward + idle replicas, the run
                            COMPLETES (a buggy solution deadlocks -> a wall-clock
                            watchdog os._exit's it -> the test fails, not hangs).
  test_no_dummy_deadlocks   without the dummy forward, an idle replica DOES wedge
                            the group (the watchdog fires -> run_on_ranks raises).

Run:
    CUDA_VISIBLE_DEVICES=0,1 uv run pytest bootcamp/tests/test_ex11_dp_lockstep.py -v
    USE_REFERENCE=1 CUDA_VISIBLE_DEVICES=0,1 uv run pytest bootcamp/tests/test_ex11_dp_lockstep.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

from bootcamp.dist_utils import require_gpus, run_on_ranks
from bootcamp.ex11_dp_lockstep.common import ToyLayer, install_watchdog, make_schedule

if os.environ.get("USE_REFERENCE"):
    from bootcamp.ex11_dp_lockstep.reference import dp_lockstep_forward, lockstep_tax
else:
    from bootcamp.ex11_dp_lockstep.solution import dp_lockstep_forward, lockstep_tax

# Small + fast: the point is the collective schedule, not throughput.
HIDDEN = 512
FFN_MULT = 2
LAYERS = 4
STEPS = 16
BASE = 256
EXIT_DEADLOCK = 99          # child exit code the watchdog uses on a detected hang
COMPLETE_WD_S = 40.0        # generous: only fires if a buggy solution deadlocks
DEADLOCK_WD_S = 15.0        # the expected-deadlock case; keep the test short


# ------------------------------------------------------------- tax (CPU, no GPU)
def test_lockstep_tax_balanced():
    c = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]        # perfectly balanced -> no tax
    assert lockstep_tax(c)["tax"] == pytest.approx(1.0)


def test_lockstep_tax_known():
    c = [[1.0, 3.0], [3.0, 1.0]]                  # maxr per step=[3,3]=6; sumr per dp=[4,4]->4
    r = lockstep_tax(c)
    assert r["sum_max"] == pytest.approx(6.0)
    assert r["max_sum"] == pytest.approx(4.0)
    assert r["tax"] == pytest.approx(1.5)          # uncorrelated peaks -> tax > 1


def test_lockstep_tax_ge_one():
    torch.manual_seed(0)
    r = lockstep_tax(torch.rand(8, 64).tolist())   # sum-of-maxes >= max-of-sums always
    assert r["tax"] >= 1.0 - 1e-9


# ------------------------------------------------------- invariant (needs GPUs)
def _completes_worker(rank: int, world_size: int, seed: int) -> None:
    device = f"cuda:{rank}"
    layers = [ToyLayer(HIDDEN, FFN_MULT, world_size, device) for _ in range(LAYERS)]
    # TP=1 => dp_size == world; some cells idle (idle_frac) to exercise the dummy path.
    sched = make_schedule(STEPS, world_size, BASE, imbalance=0.5, idle_frac=0.3, seed=seed)
    my_batches = [sched[s][rank] for s in range(STEPS)]

    wd = install_watchdog(COMPLETE_WD_S, lambda: os._exit(EXIT_DEADLOCK))
    per_step = dp_lockstep_forward(layers, my_batches, use_dummy=True)
    wd.cancel()

    assert len(per_step) == STEPS
    assert all(t > 0.0 for t in per_step)          # every step ran (dummy steps too)


def _deadlock_worker(rank: int, world_size: int, seed: int) -> None:
    device = f"cuda:{rank}"
    layers = [ToyLayer(HIDDEN, FFN_MULT, world_size, device) for _ in range(LAYERS)]
    sched = make_schedule(STEPS, world_size, BASE, imbalance=0.0, idle_frac=0.0, seed=seed)
    sched[0][1] = 0                                # force replica 1 idle at step 0
    my_batches = [sched[s][rank] for s in range(STEPS)]

    install_watchdog(DEADLOCK_WD_S, lambda: os._exit(EXIT_DEADLOCK))
    # use_dummy=False: replica 1 skips step 0's collective -> the world wedges ->
    # the watchdog os._exit(99)s every rank -> mp.spawn raises in the parent.
    dp_lockstep_forward(layers, my_batches, use_dummy=False)


@pytest.mark.parametrize("world_size", [2, 4])
def test_lockstep_completes(world_size: int) -> None:
    require_gpus(world_size)
    run_on_ranks(world_size, _completes_worker, 0)


def test_no_dummy_deadlocks() -> None:
    require_gpus(2)
    from torch.multiprocessing import ProcessExitedException
    # A real deadlock => the watchdog os._exit(99)s => spawn raises with that exact
    # code. (An unfilled/erroring solution raises ProcessRaisedException instead, so
    # this test only passes when the group genuinely wedges.)
    with pytest.raises(ProcessExitedException) as ei:
        run_on_ranks(2, _deadlock_worker, 0)
    assert ei.value.exit_code == EXIT_DEADLOCK
