# Exercise 10 — Fused hybrid: Ex09's Triton kernel inside Ex07's HybridBlock

**Goal**: Take the working `HybridBlock` from Ex07 (TP-4 × DP-2 × EP-8 on 8 GPUs,
or the 2-node TP-8 × DP-2 × EP-16 variant on 16 GPUs) and **replace its Python
per-expert loop with a single call into `fused_moe_forward` from Ex09**. Same
end-to-end contract, faster local MoE compute.

This is the **first real composition** in the microbenchmark suite — two
previously-verified components (`ex07.HybridBlock`, `ex09.fused_moe_forward`)
snapped together. It's also the pilot experiment for the paper's
agent-composition arm: we do this composition manually first to establish a
correctness + performance ground truth, then use it to score agent attempts on
the same task.

## What changes vs. Ex07

Ex07's `HybridMoE.forward` at
[bootcamp/ex07_tp_ep_hybrid/solution.py:205-211](../ex07_tp_ep_hybrid/solution.py#L205-L211)
runs a Python loop over local experts:

```python
expert_ids_local_T_sorted_cnts = torch.bincount(expert_ids_local_T_sorted,
                                                minlength=self.experts_per_rank)
start = 0
for idx, cnt in enumerate(expert_ids_local_T_sorted_cnts.tolist()):
    if cnt == 0:
        continue
    y_local_T_sorted[start:start+cnt] = self.experts[idx](x_local_T_sorted[start:start+cnt])
    start += cnt
```

Ex10 replaces this with:

```python
counts = torch.bincount(expert_ids_local_T_sorted, minlength=self.experts_per_rank)
offsets = F.pad(counts.cumsum(0), (1, 0)).to(torch.int64)  # [E_per_rank + 1]
y_local_T_sorted = fused_moe_forward(
    x_local_T_sorted, offsets,
    self.W_gate_packed, self.W_up_packed, self.W_down_packed,
)
```

plus an `__init__`-time pack of the `nn.ModuleList` of `RefSwiGLU_MLP` into
contiguous `[E_per_rank, I, H]` / `[E_per_rank, H, I]` weight tensors via
`bootcamp.ex09_fused_moe.reference.pack_expert_weights`.

## Correctness contract (the composition claim)

For every routing decision produced by the router, on every rank, the output
of Ex10's `HybridBlock.forward(x)` equals Ex07's `HybridBlock.forward(x)` up
to declared floating-point reduction-order tolerance. The test suite pins
`atol=1e-4, rtol=1e-4` for fp32 and `atol=5e-2, rtol=5e-2` for bf16 — the same
tolerances Ex07 and Ex09 use individually.

The composition proof (see the paper's `theorem` section) reduces to two
obligations, both discharged by the individual Verus contracts:
1. `ex09.fused_moe_forward`'s postcondition matches the per-expert-loop
   postcondition inside `HybridMoE`.
2. `ex07.HybridMoE`'s pre-condition on `(x_local_T_sorted, offsets)` matches
   `ex09.fused_moe_forward`'s precondition.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex10_fused_moe_hybrid.py -v
```

The test compares Ex10 (fused) vs Ex07 (Python loop) on the same inputs,
same seed, same routing.

## Two-node run

```sh
bash bootcamp/ex10_fused_moe_hybrid/launch_multinode.sh
```

Reuses Ex07's launcher, uses `NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=net1-0`.

## File layout

- **`reference.py`** — the manual composition (working, tests pass).
  Serves the usual pedagogical role AND the paper's ground-truth role.
  Run tests against it via `USE_REFERENCE=1 pytest ...`.
- **`solution.py`** — currently a stub. The pilot-experiment target:
  the coding agent is asked to produce a working `FusedHybridBlock`
  here, given (i) Ex07's `solution.py`, (ii) Ex09's `reference.py`,
  and (iii) the Verus contracts for both. Scored against `reference.py`.

## Paper role

- **Manual composition** (Wei, `reference.py`): the ground-truth
  reference. Ships in the artifact.
- **Verus contract** (Jun): chains Ex07's and Ex09's contracts into the
  composition proof; ~30 lines.
- **Agent-pilot experiment** (Wei + Claude): agents given Ex07's solution
  + Ex09's reference + the Verus contracts must produce a matching
  composition into `solution.py`. Scored against `reference.py`.
