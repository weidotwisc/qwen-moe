"""[C7/C8 step 2] Parallel state: the TP/EP device mesh, as module globals.

nano-vLLM was a flat world (`tp == ep == world_size`, one default group). This
splits the world into a mesh:

  world = tensor_parallel_size (TP) * data_parallel_size (DP) = #GPUs = ep_size

  - `tp_group` : one per DP replica, `TP` contiguous ranks; attention / dense
                 linears / embedding / LM head, and the MoE stripe + all_gather.
  - `ep_group` : the whole world; experts shard across ALL ranks, so this is the
                 default group (`None`) -- only the MoE dispatch/combine use it.

Layers call the accessors below instead of `dist.get_world_size()/get_rank()`, so
the group they shard/reduce over is decided here (set once by `init_parallel` in
model_runner, before the model is built) rather than hardcoded to the world.

`DP == 1` leaves `tp_group = None` (the default world group), so every accessor
returns exactly what `dist.*()` returned before this module existed -- the mesh is
byte-identical to the old flat path at DP=1. (DP>1 builds the group STRUCTURE; the
batch is still replicated across replicas until the engine-level DP of step 3 --
so DP>1 here is redundant, not yet a speedup.)
"""
import torch.distributed as dist

# This rank's groups. None == the default (world) group. Set by init_parallel().
_TP_GROUP = None
_EP_GROUP = None
_TP_RANKS: list[int] | None = None   # global ranks in this rank's tp_group
# [DP step 3] the DP-leader group = the tp_rank-0 rank of every replica ([0, TP, 2TP, …]);
# used to gather each replica's sampled tokens back to rank 0. None when DP==1.
_DP_LEADER_GROUP = None
_REPLICA_INDEX = 0                    # this rank's DP replica index (rank // tp_size)


def init_parallel(tp_group, ep_group, tp_ranks: list[int],
                  dp_leader_group=None, replica_index: int = 0) -> None:
    """Install the mesh for this process. Call once, after init_process_group and
    before constructing the model (layers read the accessors at __init__)."""
    global _TP_GROUP, _EP_GROUP, _TP_RANKS, _DP_LEADER_GROUP, _REPLICA_INDEX
    _TP_GROUP, _EP_GROUP, _TP_RANKS = tp_group, ep_group, tp_ranks
    _DP_LEADER_GROUP, _REPLICA_INDEX = dp_leader_group, replica_index


def get_tp_group():
    return _TP_GROUP


def get_ep_group():
    return _EP_GROUP


def get_tp_world_size() -> int:
    return dist.get_world_size(_TP_GROUP) if dist.is_initialized() else 1


def get_tp_rank() -> int:
    return dist.get_rank(_TP_GROUP) if dist.is_initialized() else 0


def get_ep_world_size() -> int:
    return dist.get_world_size(_EP_GROUP) if dist.is_initialized() else 1


def get_ep_rank() -> int:
    return dist.get_rank(_EP_GROUP) if dist.is_initialized() else 0


def get_tp_leader_global_rank() -> int:
    """Global rank of tp_rank 0 in this rank's tp_group (the LM-head gather dst)."""
    return _TP_RANKS[0] if _TP_RANKS else 0


def get_dp_leader_group():
    """[DP step 3] Process group over the DP-replica leaders ([0, TP, 2TP, …]); None at DP==1."""
    return _DP_LEADER_GROUP


def get_replica_index() -> int:
    """[DP step 3] This rank's DP replica index (rank // tp_size)."""
    return _REPLICA_INDEX


def is_tp_leader() -> bool:
    """[DP step 3] True on the tp_rank-0 rank of each replica (the sampler / gather participant)."""
    return get_tp_rank() == 0
