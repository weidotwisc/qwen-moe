# Ex11 — the DP lockstep invariant + the "lockstep tax"

An exercise **and** a `torchrun` micro-benchmark (no nano-vLLM import) that distil
the one fully transferable idea from nano-vLLM's central-orchestration data
parallelism (engine step 3, commit `8734e92`). Fill in
[solution.py](solution.py) (two functions — `dp_lockstep_forward` and
`lockstep_tax`) and check it against [reference.py](reference.py) with the tests:

```sh
# your solution (fails/deadlocks if the invariant is wrong):
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run pytest bootcamp/tests/test_ex11_dp_lockstep.py -v
# reference sanity check:
USE_REFERENCE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 uv run pytest bootcamp/tests/test_ex11_dp_lockstep.py -v
```

Scaffolding (ToyLayer, the all-to-all barrier, schedule, mesh) is given in
[common.py](common.py). The test converts a buggy/deadlocking solution into a
clean failure (a wall-clock watchdog `os._exit`s a wedged rank) rather than a hang.

## The idea

nano-vLLM runs expert parallelism with `ep_group == world`, so a cross-replica
`all_to_all` sits in **every** MoE forward. Consequence: the DP replicas are
**not** independent DDP workers — they must **co-arrive at every layer**, or the
world-wide collective blocks forever. A replica whose scheduler is momentarily
empty *cannot* just skip a step; it must run a **dummy forward** — a token-less
forward whose only purpose is to post its share of the collective schedule.

Real reference: the MoE block posts **3 all-to-alls per layer** over
`ep_group=world` (split-negotiate + dispatch + combine),
[ex06_ep_pure/reference.py:117-153](../ex06_ep_pure/reference.py#L117-L153).
This toy keeps the load-independent split-negotiation `all_to_all` (a tiny
`[world]` exchange) as the co-arrival barrier and folds the load-dependent work
(expert compute + dispatch/combine volume) into an O(n) MLP — so the tax lands on
the compute where the batch size actually lives.

## (1) The invariant — a real deadlock

```
./run.sh invariant
```

- **dummy forward** (default): the idle replica posts a token-less collective →
  the group **completes**.
- **`--no-dummy`**: the idle replica skips the collective → the busy replicas
  block on the world `all_to_all` → **DEADLOCK**, caught by a wall-clock watchdog
  that self-reports and hard-exits the wedged process (a self-`pkill`, but with a
  verdict — a NCCL collective blocks in the CUDA stream, so this cannot be caught
  with `try/except`).

Observed (8×A100, TP=1 DP=8):

```
[1/2] dummy    → RESULT: COMPLETED — every replica posted the collective schedule.
[2/2] no-dummy → RESULT: DEADLOCK CONFIRMED — no progress for 15s.
                 idle replica r7 never posted the all_to_all; busy ranks blocked.
```

This is the crisp control-plane failure mode: correctness lives in *who posts
which collective when*, not in any kernel.

## (2) The lockstep tax

Because every replica co-arrives per step, the **slowest replica gates the whole
step**. Define per-step per-replica compute `c[s,r]`:

- **ideal** (independent DP, no cross-replica sync): `wall = max_r Σ_s c[s,r]`
- **lockstep** (what `ep_group=world` forces):        `wall = Σ_s max_r c[s,r]`

`tax = lockstep / ideal ≥ 1`, with equality only when the same replica is the
bottleneck every step. The gap is driven by **per-step load imbalance** and the
**idle→dummy fraction** — quantities we could not otherwise guess.

```
./run.sh tax          # sweep imbalance × idle_frac, JSONL → results/
```

**Result (8×A100, TP=1 DP=8, steps=64 layers=16 H=2048 base=2048 tok/replica):**

| imbalance | idle_frac | tax (structural) | tax (wall) | a2a ovh | lockstep tok/s | ideal tok/s |
|----------:|----------:|-----------------:|-----------:|--------:|---------------:|------------:|
| 0.0 | 0.0  | **1.00×** | 1.02× | 1.7% | 2.44M | 2.49M |
| 0.0 | 0.25 | 1.17× | 1.20× | 2.4% | 1.90M | 2.28M |
| 0.0 | 0.5  | 1.44× | 1.48× | 2.4% | 1.23M | 1.82M |
| 0.25 | 0.0  | 1.19× | 1.21× | 0.9% | 2.14M | 2.58M |
| 0.25 | 0.25 | 1.40× | 1.42× | 1.4% | 1.68M | 2.38M |
| 0.25 | 0.5  | 1.58× | 1.61× | 1.8% | 1.15M | 1.85M |
| 0.5 | 0.0  | 1.50× | 1.50× | 0.0% | 1.78M | 2.67M |
| 0.5 | 0.25 | 1.68× | 1.69× | 0.6% | 1.42M | 2.40M |
| 0.5 | 0.5  | 1.84× | 1.86× | 1.4% | 1.01M | 1.88M |
| 1.0 | 0.0  | 1.93× | 1.92× | 0.0% | 1.42M | 2.73M |
| 1.0 | 0.25 | 2.13× | 2.12× | 0.0% | 1.15M | 2.43M |
| 1.0 | 0.5  | 2.06× | 2.07× | 0.7% | 0.87M | 1.81M |

Perfect balance ⇒ **1.00×** (no tax — sanity check). The tax then climbs
monotonically with both knobs to **~2.1×**: central-lockstep DP leaves up to
half the cluster idle-waiting on the slowest replica per step. This is the cost
`ep_group=world` buys in exchange for making cross-replica EP possible at all —
a number we previously could not guess.

`tax_structural` is the pure gating cost (`Σmax / maxΣ`, artifact-free);
`tax_wall` is the measured end-to-end ratio; `a2a_ovh%` is the load-independent
negotiation collective's share of the lockstep wall (small — the barrier is
cheap; the tax is structural).

## Paper hook

The lockstep rule is a clean **control-plane property**: *every rank posts an
identical collective schedule per step*. Its violation is a real hang, and its
cost (the tax) is a real, previously-unquantified number. Sits next to the
KV-cache OOB bug (`9dbe261`) as evidence that the **control plane, not the
kernels**, is where the subtle correctness — and now the subtle cost — lives.
Candidate motivating example for the verified-control-plane thread.

## Files

- `solution.py` — **fill this in**: `dp_lockstep_forward` (the invariant loop) +
  `lockstep_tax` (Σmax / maxΣ).
- `reference.py` — the answer key for those two functions.
- `common.py` — given scaffolding: `ToyLayer`, `run_layers`, `make_schedule`,
  `build_mesh`, `reduce_max_scalar`, `install_watchdog`.
- `../tests/test_ex11_dp_lockstep.py` — tax (CPU) + completes/deadlock (GPU) tests.
- `dp_lockstep_bench.py` — the bench (`--mode invariant | tax`), torchrun entry;
  shares `common.py` + `reference.py`.
- `run.sh` — driver: `invariant` | `tax` | `all` (default). Env: `NPROC` (8),
  `TP_SIZE` (1 → DP=NPROC).
- `results/tax_tp*_dp*.jsonl` — sweep output.

## Cleanup

A wedged run: `pkill -9 -u "$USER" -f dp_lockstep_bench`.
