# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlyDSL-backed GEMM kernels for MSLK (ROCm / gfx942, gfx950)."""

from mslk.gemm.flydsl.preshuffle_gemm import (
    f8f8bf16_rowwise_preshuffle,
    f8f8f16_rowwise_preshuffle,
)

# Importing `register` binds FlyDSL as the CUDA impl of the mslk:: preshuffle
# ops as a side effect (see register.py); `register` is re-exported so callers
# can retry the binding if the C++ schema was not yet loaded at import time.
from mslk.gemm.flydsl.register import register  # noqa: E402

__all__ = [
    "f8f8bf16_rowwise_preshuffle",
    "f8f8f16_rowwise_preshuffle",
    "register",
]
