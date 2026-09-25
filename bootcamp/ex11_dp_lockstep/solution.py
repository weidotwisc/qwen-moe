"""Ex11 SOLUTION — fill in the DP lockstep step + the lockstep tax.

The one idea: nano-vLLM's DP engine runs `ep_group == world`, so a cross-replica
all_to_all sits in EVERY MoE forward. The DP replicas are therefore NOT independent
DDP workers -- they must co-arrive at every layer, driven by a central controller
as one synchronized global step. Skip a step on one replica and the whole world
hangs on a collective that never completes.

Fill in the two functions below, then:

    # test YOUR solution (fails/deadlocks if the invariant is wrong):
    CUDA_VISIBLE_DEVICES=0,1 uv run pytest bootcamp/tests/test_ex11_dp_lockstep.py -v

    # sanity-check the reference passes the same tests:
    USE_REFERENCE=1 CUDA_VISIBLE_DEVICES=0,1 uv run pytest bootcamp/tests/test_ex11_dp_lockstep.py -v

Scaffolding (ToyLayer, run_layers, make_schedule, mesh) is given in common.py --
you only implement the two cores. See reference.py once you want the answer key.
"""

from __future__ import annotations

import time  # noqa: F401  (you'll want time.perf_counter for per-step timing)

import torch  # noqa: F401  (torch.cuda.synchronize / torch.as_tensor)
import torch.distributed as dist  # noqa: F401

from bootcamp.ex11_dp_lockstep.common import ToyLayer, run_layers  # noqa: F401


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

    HINTS:
      - `run_layers(layers, n, collective=True)` runs all layers for n tokens and
        posts the world all_to_all once per layer. It maps n<=0 to a 1-row dummy,
        so calling it with n==0 is exactly the "dummy forward".
      - THE INVARIANT: every replica must post the same collective schedule every
        step. Under what condition is it correct to NOT call run_layers this step?
        (Answer: only the buggy use_dummy=False + idle case -- and that deadlocks.)
      - Time each real step with time.perf_counter() around a torch.cuda.synchronize().
    """
    raise NotImplementedError("Ex11: implement dp_lockstep_forward (the lockstep invariant)")


def lockstep_tax(c) -> dict:
    """The lockstep tax from a per-step per-replica compute matrix.

    Args:
        c: shape [dp, steps] -- c[r][s] is replica r's compute at step s.

    Returns dict(sum_max, max_sum, tax) where:
        ideal    (independent DP, no cross-replica sync): wall = max_r Sum_s c[r,s]
        lockstep (ep_group=world forces co-arrival):      wall = Sum_s max_r c[r,s]
        tax = lockstep / ideal  >= 1.

    HINTS:
      - torch.as_tensor(c, dtype=torch.float64) gives a [dp, steps] tensor.
      - lockstep gates on the slowest replica PER STEP: max over dim=0, then sum.
      - ideal is each replica's own total, then the worst: sum over dim=1, then max.
    """
    raise NotImplementedError("Ex11: implement lockstep_tax (sum-of-maxes / max-of-sums)")
