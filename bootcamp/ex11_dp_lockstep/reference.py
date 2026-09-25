"""Ex11 REFERENCE — the DP lockstep step + the lockstep tax.

The working answer key. `solution.py` has the same two functions for you to fill;
tests import from here when USE_REFERENCE=1, else from the solution.

Both functions capture the one transferable idea from nano-vLLM's DP engine
(commit 8734e92): because `ep_group == world`, a cross-replica all_to_all sits in
every forward, so the DP replicas must **co-arrive at every layer**. They are not
independent DDP workers -- a central controller drives one synchronized global
step across all of them.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from bootcamp.ex11_dp_lockstep.common import ToyLayer, run_layers


def dp_lockstep_forward(
    layers: list[ToyLayer], my_batches: list[int], *, use_dummy: bool
) -> list[float]:
    """Run this replica's stream of per-step batches in lockstep with the world.

    Args:
        layers:      this replica's ToyLayers (each posts a world all_to_all).
        my_batches:  per-step token counts for THIS replica (0 == idle this step).
        use_dummy:   if True, an idle step still runs a (token-less) forward so the
                     replica posts its share of the collective. If False, an idle
                     step is skipped entirely -- the WRONG behavior that deadlocks.

    Returns:
        per-step wall time on this rank (0.0 for a skipped idle step).

    THE INVARIANT: every replica must post the same collective schedule every
    step. `run_layers(..., collective=True)` posts the world all_to_all once per
    layer; a replica that skips it (idle + not use_dummy) leaves the rest of the
    world blocking on a collective that will never complete.
    """
    per_step: list[float] = []
    for n in my_batches:
        if n == 0 and not use_dummy:
            # BUG PATH: the scheduler is empty this step, so we return early and
            # never post the all_to_all. The busy replicas block on it forever.
            per_step.append(0.0)
            continue
        t0 = time.perf_counter()
        run_layers(layers, n, collective=True)   # n==0 -> 1-row dummy, still posts
        torch.cuda.synchronize()
        per_step.append(time.perf_counter() - t0)
    return per_step


def lockstep_tax(c) -> dict:
    """The lockstep tax from a per-step per-replica compute matrix.

    Args:
        c: shape [dp, steps] -- c[r][s] is replica r's compute at step s.

    Returns dict(sum_max, max_sum, tax) where:
        ideal  (independent DP, no cross-replica sync): wall = max_r Sum_s c[r,s]
        lockstep (ep_group=world forces co-arrival):    wall = Sum_s max_r c[r,s]
        tax = lockstep / ideal  >= 1  (equality iff the same replica gates every
        step). The gap is the cost of central lockstep vs ideal independent DP.
    """
    t = torch.as_tensor(c, dtype=torch.float64)          # [dp, steps]
    sum_max = float(t.amax(dim=0).sum())                 # Sum_s max_r  (lockstep)
    max_sum = float(t.sum(dim=1).amax())                 # max_r Sum_s  (ideal)
    return {"sum_max": sum_max, "max_sum": max_sum,
            "tax": sum_max / max_sum if max_sum else float("nan")}
