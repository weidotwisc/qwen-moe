# Ex07 — Formal properties for the TP × DP × EP hybrid block

This directory formalizes the correctness of
[`solution.py`](../solution.py) — `HybridMoE` + `HybridBlock`, the
composition of attention TP-4 (from Ex04) with MoE EP-8 (from Ex06) under
a rank layout where **three parallelism axes cross-cut on the same set of
GPUs**:

```
DP-2 (outer, no cross-replica comm during forward):
    replica_a = {0, 1, 2, 3}   replica_b = {4, 5, 6, 7}
TP-4 (per-replica, attention sub-groups):
    tp_group_a = {0, 1, 2, 3}  tp_group_b = {4, 5, 6, 7}
EP-8 (over all 8, for MoE dispatch/combine):
    ep_group = {0, 1, ..., 7}
```

Under this layout, the input to MoE is **replicated within each TP group
of 4 ranks** (attention's row-parallel all-reduce establishes this) but
**different across TP groups** (each TP group handles a different DP
replica's half of the batch). MoE dispatch must therefore cross TP-group
boundaries.

**This is the paper's centerpiece composition proof.** It composes:

1. **TP correctness** (from Ex01, Ex03, Ex04) — attention output is
   replicated within each TP group.
2. **EP correctness** (from Ex06) — dispatch/compute/combine preserves
   routing invariants.
3. **Sub-group cross-cutting deadlock-freedom** — nested TP and EP
   collectives complete without stalling each other.

**Style follows [ex06/verification/PROPERTIES.md](../../ex06_ep_pure/verification/PROPERTIES.md).**

## Abstraction model additions

Ex04's TP model and Ex06's EP model are inherited. New elements:

| Concept | Model |
|---|---|
| `tp_group_of(rank)` | `Group` — the TP sub-group that this rank belongs to. |
| `ep_group_of(rank)` | `Group` — the EP group (= all 8 ranks in our setup). |
| `dp_replica_of(rank)` | `nat` — which DP replica this rank is in. |
| `Replicated_in_tp(tensor, rank)` | Predicate: tensor is identical on all ranks in `tp_group_of(rank)`. |
| `TP_striping(tensor, rank, tp_size)` | Predicate: within a TP group, this rank holds the `tp_rank`-th 1/tp_size slice of a full tensor. |

## Properties to verify

### H1 — Attention output is TP-replicated (from Ex04)

After the attention block runs on rank r:

$$
\forall r, r' \in \text{tp\_group\_of}(r) : \text{attn\_out}_r = \text{attn\_out}_{r'}
$$

**Proof**: Ex04's R4 (RowParallelLinear all-reduce reconstructs the
o_proj matmul, replicated within the tp_group).

### H2 — MoE-input striping preserves DP replica boundary

Phase 0 of `HybridMoE.forward` splits `x_flat` into `tp_size` slices per
tp_group; rank r takes slice `tp_rank(r)`. Across DP replicas, tp_group_a
and tp_group_b see distinct halves of the batch.

$$
\forall r \in \text{replica\_a}, r' \in \text{replica\_b} : \text{local\_x}_r \ne \text{local\_x}_{r'}
$$

(equality would only hold if the two replicas process the same batch,
which contradicts DP).

**Proof**: definitional; the striping is deterministic on `(tp_rank,
dp_rank)`.

### H3 — EP dispatch crosses TP-group boundaries correctly

The all-to-all in `HybridMoE.forward` runs over `self.ep_group` (all 8
ranks), NOT over tp_group. Dispatch counts negotiated via
`all_to_all_single` on `self.ep_group` are symmetric (send matches recv)
across the full ep_group, not restricted to tp_group.

**Proof**: inherits Ex06's EP3 (dispatch symmetry) with `ep_group` as the
comm domain.

### H4 — MoE output is TP-replicated after the final all-gather

Phase 8 of `HybridMoE.forward` does
`all_gather_into_tensor(y_flat, local_y, group=self.tp_group)`, which
reconstructs the full [N, H] tensor on every rank in tp_group.

$$
\forall r, r' \in \text{tp\_group\_of}(r) : \text{moe\_out}_r = \text{moe\_out}_{r'}
$$

**Proof**: definitional postcondition of `all_gather_into_tensor` on
`tp_group`.

### H5 — Sub-group deadlock-freedom

The block issues collectives on two overlapping sub-groups (`tp_group`
and `ep_group`) in a fixed sequence. Because the sequence is
straight-line (data-independent) and every rank enters the same
sequence, no deadlock is possible.

**Proof**: syntactic property of the schedule, mirroring Ex06's EP6.

### H6 — Block correctness (composition of H1-H4 + Ex04-R4 + Ex06-EP7)

The output of `HybridBlock.forward(local_x)` on any rank equals the
per-tp-group slice of a single-GPU reference block's output, up to
`approx_eq` tolerance.

**Proof composition**:
1. H1 gives attention output replicated within tp_group.
2. H2 + H3 give correct dispatch of MoE inputs across ep_group.
3. Ex06's routing correctness (EP7) gives that local MoE compute is
   correct.
4. H4 gives that the final output is TP-replicated.
5. Composition with the residual + norm gives block-level equivalence.

Stated as a stub — the full proof is the paper's headline claim about
composition and is deferred to the composition theorem in `verus/`.

## What each tool proves — this exercise

Same three-tool pattern as Ex01-06. Verus proof shipped first.

## Correspondence to Python

1. `HybridBlock.__init__` constructs `TPGQA(...)` and `HybridMoE(...)` with
   the same `tp_group` and `ep_group` arguments. The composition follows
   the paper's abstract topology diagram directly.
2. `HybridMoE.forward` runs a fixed sequence of collectives (Phase 0-9
   as annotated). Every collective identifies its group explicitly
   (`self.ep_group` or `self.tp_group`), matching H3/H4.
3. H5 (deadlock-freedom) is a straight-line-schedule property visible by
   inspection of `HybridMoE.forward` — no data-dependent branches on
   any rank's private state.
