# qwen-moe — Expert Parallelism for nanovllm

Personal project extending [nanovllm](https://github.com/GeeeekExplorer/nano-vllm)
to support expert parallelism for Qwen MoE models. Built on the same
uv-managed Python environment as my CS336 self-study
(`cs336/assignment2-systems`) so the CUDA / Triton toolchain is already
proven.

## Environment

- **Python**: 3.13 (managed by uv).
- **Hardware target**: NVIDIA A100-SXM4-80GB, driver 550 (CUDA 12.4).
- **Torch**: 2.11.0+cu126 (via the PyTorch `cu126` wheel index).  We
  pin cu126 rather than PyPI's default cu130 because driver 550 does
  not support the r580 series that cu130 wheels require.  CUDA 12.x
  minor-version compatibility lets driver 550 run cu126 wheels.
- **Triton**: 3.6.0 (bundled with torch).
- **Transformers**: 5.14.1 (for Qwen model definitions).

## Setup (one-shot, already done)

```sh
cd /gpfs/users/weiz/workspace/personal/qwen-moe
uv init --python 3.13 --name qwen-moe
# Then edited pyproject.toml — see that file for the pinned cu126 index.
uv sync
```

`uv`'s global package cache means the second install of torch etc. was
near-instant — wheels already lived on disk from cs336-systems.

## Verify

```sh
CUDA_VISIBLE_DEVICES=1 uv run python -c "
import torch
assert torch.cuda.is_available()
assert torch.__version__.startswith('2.11.0+cu126')
print(torch.__version__, torch.cuda.get_device_name(0))
"
```

Expected output:
```
2.11.0+cu126 NVIDIA A100-SXM4-80GB
```

## Project layout

```
qwen-moe/
├── pyproject.toml       # deps + cu126 index config
├── qwen_moe/            # main package (currently empty; will host expert-parallel code)
│   └── __init__.py
├── README.md            # this file
├── .python-version      # `3.13`
└── (nanovllm/           # cloned here later, as editable-source dep)
```

## Next steps

1. **Clone nanovllm** into `./nanovllm/` and add to `pyproject.toml` under
   `[tool.uv.sources]`:
   ```toml
   nanovllm = { path = "./nanovllm", editable = true }
   ```
   Then `uv sync` again. Editable install means source edits are picked
   up without reinstalling — same pattern cs336 uses for `cs336-basics`.

2. **Sketch the expert-parallel design** for Qwen MoE:
   - How MoE experts are sharded across GPUs.
   - Communication pattern for token dispatch / combine (all-to-all
     or ring-based).
   - Where nanovllm's inference-loop hooks need to be modified.

3. **Set up a test Qwen MoE checkpoint** for correctness validation.
   Qwen2-MoE-A2.7B is a reasonable small target for iteration.

## Key packages installed

| Package | Version | Purpose |
|---|---|---|
| torch | 2.11.0+cu126 | Core |
| triton | 3.6.0 | Custom kernels if needed |
| transformers | 5.14.1 | Qwen model definitions + tokenizer |
| safetensors | 0.8.0 | Weight loading (.safetensors format) |
| tokenizers | 0.22.2 | Fast BPE / SentencePiece |
| numpy | latest | Standard |
| jsonlines | latest | Preferred for `.jsonl` I/O |
| pytest | 9.1.1 | Test runner |
| ruff | 0.16.1 | Linter |

Plus the transitive CUDA runtime deps (`nvidia-cuda-*`, `nvidia-nccl-cu12`,
`nvidia-cudnn-cu12`, etc.) that torch pulls in automatically.

## Working directory conventions

- All commands assume `cwd = /gpfs/users/weiz/workspace/personal/qwen-moe`.
- Use `uv run python ...` rather than activating the venv directly —
  keeps the `.venv/` isolated from parent shells.
- Set `CUDA_VISIBLE_DEVICES=<free_gpu>` explicitly on all runs, since
  this is a shared cluster.
