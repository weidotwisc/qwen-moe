"""Loader for the C1 CUDA fused-MoE extension (cuBLAS grouped GEMM + custom kernels)."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD = None


def load(verbose=False):
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load as _load
        _MOD = _load(
            name="moe_c1",
            sources=[os.path.join(_HERE, "moe_cuda", "moe.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_80", "--use_fast_math"],
            extra_ldflags=["-lcublas"],
            verbose=verbose,
        )
    return _MOD
