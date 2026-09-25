"""Ex11 shared infra — GIVEN. The exercise core lives in reference.py / solution.py.

This module holds the toy MoE-DP machinery so the exercise can focus on the two
things worth understanding: the lockstep step loop (`dp_lockstep_forward`) and the
lockstep tax (`lockstep_tax`). Nothing here is part of the fill-in.

Model: a TP x DP mesh over `world` GPUs with `ep_group == world`. Each "layer" is
an O(n) expert MLP (the per-replica load) followed by the load-independent
split-negotiation all_to_all over the whole world (ex06 Phase 4) -- the co-arrival
barrier every replica must post.
"""

from __future__ import annotations

import os
import threading
from collections import namedtuple

import torch
import torch.distributed as dist

Mesh = namedtuple("Mesh", "rank world local_rank tp_size dp_size tp_group replica tp_rank")


def build_mesh() -> Mesh:
    """Read torchrun env, set the device, build contiguous TP groups (created in
    identical order on every rank -- new_group is itself a collective), return the
    mesh. ep_group is the whole world, so it is not returned (pass group=None)."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    tp_size = int(os.environ.get("TP_SIZE", 1))
    assert world % tp_size == 0, f"world({world}) not divisible by TP_SIZE({tp_size})"
    dp_size = world // tp_size

    torch.cuda.set_device(local_rank)
    tp_groups = [
        dist.new_group(ranks=list(range(g * tp_size, (g + 1) * tp_size)))
        for g in range(dp_size)
    ]
    replica = rank // tp_size
    tp_rank = rank % tp_size
    return Mesh(rank, world, local_rank, tp_size, dp_size, tp_groups[replica], replica, tp_rank)


class ToyLayer:
    """One 'MoE layer': an O(n) expert MLP (the per-replica load) + the world
    all_to_all that forces co-arrival (modelled as the load-independent [world]
    split-negotiation exchange -- ex06 Phase 4)."""

    def __init__(self, hidden: int, ffn_mult: int, world: int, device, dtype=torch.bfloat16):
        f = hidden * ffn_mult
        self.W1 = torch.randn(hidden, f, device=device, dtype=dtype) / (hidden ** 0.5)
        self.W2 = torch.randn(f, hidden, device=device, dtype=dtype) / (f ** 0.5)
        self.hidden = hidden
        self.nego_in = torch.zeros(world, dtype=torch.int64, device=device)
        self.nego_out = torch.empty(world, dtype=torch.int64, device=device)
        self.device = device
        self.dtype = dtype

    def compute(self, n: int):
        x = torch.randn(max(n, 1), self.hidden, device=self.device, dtype=self.dtype)
        _ = torch.relu(x @ self.W1) @ self.W2                # O(n): expert MLP

    def barrier(self):
        dist.all_to_all_single(self.nego_out, self.nego_in)  # over world (ep_group)


def run_layers(layers: list[ToyLayer], n: int, *, collective: bool):
    """Run every layer for a batch of `n` tokens (n<=0 -> a 1-row dummy). When
    `collective`, post the world all_to_all after each layer -- this is the
    co-arrival barrier the whole world must reach in lockstep."""
    n_eff = max(n, 1)
    for layer in layers:
        layer.compute(n_eff)
        if collective:
            layer.barrier()


def make_schedule(steps, dp_size, base_tokens, imbalance, idle_frac, seed) -> list[list[int]]:
    """steps x dp_size token counts, identical on every rank (the central controller
    broadcasts one schedule). iid per (step,replica): idle w.p. idle_frac, else
    base*(1 + imbalance*N(0,1)) clamped to >=1."""
    g = torch.Generator().manual_seed(seed)
    sched = []
    for _ in range(steps):
        row = []
        for _ in range(dp_size):
            if torch.rand(1, generator=g).item() < idle_frac:
                row.append(0)
            else:
                z = torch.randn(1, generator=g).item()
                row.append(max(1, round(base_tokens * (1.0 + imbalance * z))))
        sched.append(row)
    return sched


def reduce_max_scalar(x: float, device) -> float:
    """all_reduce MAX of a scalar -> the slowest rank's value (the true wall)."""
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def install_watchdog(seconds: float, on_timeout):
    """Fire `on_timeout` after `seconds` of no progress. A NCCL collective blocks
    in the CUDA stream, so a genuine hang cannot be caught with try/except -- this
    wall-clock timer is the only safe detector. Cancel it once the work completes."""
    t = threading.Timer(seconds, on_timeout)
    t.daemon = True
    t.start()
    return t
