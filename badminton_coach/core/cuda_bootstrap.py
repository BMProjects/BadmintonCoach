"""Make onnxruntime-gpu find the CUDA libs that torch ships in site-packages.

onnxruntime-gpu's CUDA provider dlopen's libcublasLt/libcublas/libcudnn by soname;
without LD_LIBRARY_PATH it fails (`libcublasLt.so.12: cannot open shared object`)
and silently falls back to CPU. torch (cu12x wheels) already bundles these under
site-packages/nvidia/*/lib, so we preload them globally before the ORT session is
built. Call preload_cuda_libs() before creating any CUDA onnxruntime session.
"""

from __future__ import annotations

import ctypes
import glob
import os

_done = False

# Order matters: dependencies (cublas) before dependents (cublasLt), cudnn last.
_LIBS = (
    "cuda_runtime/lib/libcudart.so*",
    "cublas/lib/libcublas.so*",
    "cublas/lib/libcublasLt.so*",
    "cufft/lib/libcufft.so*",
    "curand/lib/libcurand.so*",
    "cudnn/lib/libcudnn*.so*",
)


def preload_cuda_libs() -> bool:
    """Preload torch's bundled CUDA libs globally. Idempotent. Returns True if run."""
    global _done
    if _done:
        return True
    try:
        import nvidia  # provided by torch's nvidia-* wheel deps
    except ImportError:
        return False
    base = os.path.dirname(nvidia.__file__)
    for pattern in _LIBS:
        for path in sorted(glob.glob(os.path.join(base, pattern))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    _done = True
    return True
