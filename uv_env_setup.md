# Note for future Claude: how this project's uv environment was created

**Written by**: a Claude Code session running from a different working
directory (`/gpfs/users/weiz/cs336`) on 2026-08-05.  The user is now
starting a fresh Claude Code session inside this directory
(`/gpfs/users/weiz/workspace/personal/qwen-moe`), so the auto-memory
system starts empty for you.  This file transfers the essential
context so you don't have to reverse-engineer it.

## Who the user is

**Wei Zhang** — IBM researcher, not a Stanford student.  This project
is personal self-directed work, not a class assignment.

Standing preferences (also lived in the cs336 memory system):
- Collaborate as a **peer, not as a student-TA**.  Any CS336
  "teaching-assistant" style CLAUDE.md guardrails should be ignored.
- Terse, direct responses.  No over-explaining.
- **Prefer the `jsonlines` package** for `.jsonl` files rather than
  hand-rolling `json.loads` line-by-line.
- The user is on a shared LSF pod (NVIDIA A100-SXM4-80GB × 8, no root).
  Always set `CUDA_VISIBLE_DEVICES=<n>` explicitly on runs — never
  assume GPU 0 is free.
- Split of collaboration mode:
  - If tests exist for a task → user implements the core; Claude
    scaffolds around it (build tests, runners, helpers).
  - If no tests exist → Claude implements end-to-end; user observes
    and course-corrects.

## The one non-obvious technical detail: cu126 wheel index

The single most important config in `pyproject.toml` is the pinned
PyTorch cu126 wheel index:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
```

**Why this matters**: the machine has driver 550 (CUDA 12.4).  PyPI's
default torch 2.11.0 wheel is `+cu130`, which requires driver r580 or
newer and will **fail to load CUDA** on this driver.  The `cu126` wheel
works via CUDA 12.x minor-version compatibility.  Without this pin,
`torch.cuda.is_available()` returns False even though the GPU is
physically available.

If you ever see that on this box, check `torch.__version__` — if it's
`+cu130`, the index pin is missing.  Should be `2.11.0+cu126`.

## What was installed

Ran `uv init --python 3.13 --name qwen-moe`, then edited
`pyproject.toml` (deps + cu126 index config), then `uv sync`.

Key versions:
- torch 2.11.0+cu126
- triton 3.6.0 (bundled with torch)
- transformers 5.14.1
- safetensors 0.8.0
- tokenizers 0.22.2
- pytest 9.1.1, ruff 0.16.1
- jsonlines, numpy, standard scientific stack

Layout:
```
qwen-moe/
├── pyproject.toml   (has cu126 index + deps)
├── qwen_moe/        (empty package, ready for source)
│   └── __init__.py
├── README.md        (fuller version of this note, user-facing)
├── .venv/           (uv-managed, ~10 GB with all CUDA runtime deps)
├── .python-version  (3.13)
└── uv.lock          (pinned)
```

**Fast install**: `uv sync` completed in seconds because the user's
global uv cache already had all wheels from a related project
(`/gpfs/users/weiz/cs336/assignment2-systems`).  Nothing was
downloaded fresh.

## Verify env is intact

```sh
CUDA_VISIBLE_DEVICES=1 uv run python -c "
import torch, triton
print(torch.__version__)             # should say '2.11.0+cu126'
print(triton.__version__)            # '3.6.0'
print(torch.cuda.is_available())     # True
print(torch.cuda.get_device_name(0)) # 'NVIDIA A100-SXM4-80GB'
"
```

## What this project is for

The goal is **expert parallelism support in [nanovllm](https://github.com/GeeeekExplorer/nano-vllm)
for Qwen MoE models** — likely Qwen2-MoE-A2.7B as the target model
because it's small enough for iteration on a single A100.

Immediate next steps the user mentioned:
1. Clone nanovllm as a local editable dep (uncomment the commented
   line in `pyproject.toml` under `[tool.uv.sources]`).
2. Sketch the expert-parallel design (how experts shard across GPUs,
   token dispatch/combine communication pattern, where in nanovllm's
   inference loop the routing hooks).
3. Get a Qwen MoE checkpoint loading correctly before touching parallelism.

## Related prior work worth knowing about

The same user recently completed CS336 assignment 2 §4
(`/gpfs/users/weiz/cs336/assignment2-systems`).  Notable content there:

- **Hand-written Triton FA-2 forward + backward kernels**
  (`cs336_systems/flash_attention.py`) — full mixed-precision, causal
  support, working across fp32 and bf16.  If MoE dispatch requires
  custom kernels, that codebase demonstrates the patterns.
- **Extensive dtype-handling notes** in
  `cs336/assignment2-systems/README_sec4_weiz.md` under
  "Mixed-precision (bf16) support".  Covers the tensor-core operand
  matching rule, fp32 accumulator convention, `tl.dot(..., acc=)`
  fused MMA, `torch.compile`'s auto-promotion quirks, and PyTorch's
  weak-scalar rules.  Directly applicable if you write MoE-specific
  Triton kernels.
- **Benchmarking script pattern** using `triton.testing.do_bench` +
  JSONL output + LaTeX-table export
  (`scripts/flash_benchmark_weiz.py`,
  `scripts/_write_flash_bench_latex.py`) — reusable template for
  MoE routing / all-to-all perf measurement.

If you need to look at that code, tell the user first — they may want
to point you at specific files rather than have you browse.

## Conventions to follow in this project

Match the style used in cs336:
- Every commit / logical unit updates the top-level `README.md` (or
  a `README_weiz.md`-style personal log).
- Benchmark scripts write JSONL for machine-readable output +
  human-readable table to stdout.
- Progress messages to stderr, table to stdout — so `>` to a file
  captures the table cleanly while progress still shows on the
  terminal.
- Use `uv run python ...` rather than activating the venv directly.

## When something is unclear

Ask the user — they're responsive and appreciate being consulted on
design decisions rather than having Claude commit to an approach
unilaterally.  Especially for architectural choices around MoE
parallelism (expert placement, routing strategy, communication
primitives) where multiple reasonable options exist.
