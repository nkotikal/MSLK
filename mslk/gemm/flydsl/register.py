# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bind the FlyDSL FP8 rowwise-preshuffle kernels to the ``mslk::`` op names.

This is the "Replaces" half of WP-G3 Phase A. It follows the exact pattern
MSLK already uses for Python-implemented ROCm GPU ops (see
``mslk/gemm/triton/int8_gemm.py`` for ``i8i8bf16``): the op schema is declared
in C++ (``m.def``) with NO CUDA ``m.impl``, and the CUDA implementation is
registered here from Python. Accordingly, ``csrc/gemm/gemm_ops.cpp`` no longer
registers a CK ``m.impl`` for these two ops.
"""

import warnings

import torch

from mslk.gemm.flydsl.preshuffle_gemm import (
    f8f8bf16_rowwise_preshuffle as _bf16_impl,
    f8f8f16_rowwise_preshuffle as _f16_impl,
)

_registered = False


def _register_rocm_ops() -> None:
    """Register the FlyDSL CUDA impls against the mslk preshuffle op schemas.

    Requires ``torch.ops.mslk`` to already carry the schemas (declared by the
    mslk C++ library's ``m.def``). PyTorch HIP builds surface GPU kernels under
    the "CUDA" dispatch key.
    """
    torch.library.impl(
        "mslk::f8f8bf16_rowwise_preshuffle", "CUDA"
    )(_bf16_impl)
    torch.library.impl(
        "mslk::f8f8f16_rowwise_preshuffle", "CUDA"
    )(_f16_impl)


def register() -> None:
    """Idempotently bind FlyDSL as the CUDA impl of the preshuffle ops.

    No-op on non-HIP builds. If binding fails because a conflicting CUDA impl
    (e.g. a stale CK ``m.impl``) is still registered, this raises rather than
    silently leaving the old implementation in place — a silent fallback would
    mean the migration did not take effect.
    """
    global _registered
    if _registered or torch.version.hip is None:
        return
    _register_rocm_ops()
    _registered = True


# Register on import — mslk/gemm/__init__.py loads the C++ schema before it
# imports this package, so the op schemas exist by the time we bind.
if torch.version.hip is not None:
    try:
        register()
    except Exception as exc:  # noqa: BLE001
        # Schema not loaded yet (e.g. a stubbed unit-test environment). Defer
        # to an explicit register() call by the caller.
        warnings.warn(
            f"Deferring FlyDSL preshuffle registration until later import: {exc}",
            stacklevel=2,
        )
