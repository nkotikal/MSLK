# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dev/test-only bootstrap for the FlyDSL preshuffle GEMM tests.

Everything here reproduces preconditions that a real MSLK build provides and
that the shipped modules therefore assume — kept OUT of the shipped code:

  1. In production the mslk C++ library declares the op schemas via ``m.def``.
     This repo's sandbox is the ``python_only`` mslk variant (no compiled
     csrc), so we declare the two preshuffle schemas here — exactly the schema
     from ``csrc/gemm/gemm_ops.cpp`` — before importing ``mslk.gemm`` so its
     Python registration can bind the FlyDSL CUDA impl.

  2. In production FlyDSL is an installed dependency (the ``flydsl`` package and
     the ``kernels`` template module). Here FlyDSL is only a sibling checkout,
     so we put it on ``sys.path``.
"""

import os
import sys

import torch
from torch.library import Library

# 1. Make the FlyDSL `flydsl` + `kernels` packages importable (dev checkout).
_FLYDSL_ROOT = os.environ.get("FLYDSL_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "flyDSL")
)
if os.path.isdir(_FLYDSL_ROOT) and _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

# 2. Declare the op schemas the mslk C++ library would provide in a real build.
_SIG = (
    "(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, "
    "Tensor? bias=None, bool use_fast_accum=True) -> Tensor"
)

# Held at module scope so the schema registration persists for the session.
_LIB = Library("mslk", "FRAGMENT")


def _op_missing(name: str) -> bool:
    try:
        getattr(torch.ops.mslk, name)
        return False
    except (AttributeError, RuntimeError):
        return True


for _name in ("f8f8bf16_rowwise_preshuffle", "f8f8f16_rowwise_preshuffle"):
    if _op_missing(_name):
        _LIB.define(f"{_name}{_SIG}")
