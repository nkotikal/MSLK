# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical-match test for the FlyDSL-backed FP8 rowwise preshuffle GEMM.

Drives the wrapper through the exact CK op contract (WQ pre-shuffled with
ck_preshuffle) and compares against the torch dequant reference that the CK
kernel also targets.

Run:  cd test && python -m pytest -v gemm/flydsl_preshuffle_test.py
"""

import unittest

import torch

# Importing mslk.gemm triggers registration of the FlyDSL kernels as the
# implementation of torch.ops.mslk.f8f8{bf16,f16}_rowwise_preshuffle.
import mslk.gemm  # noqa: F401
from mslk.quantize.shuffle import ck_preshuffle
from mslk.quantize.triton.fp8_quantize import quantize_fp8_row
from mslk.gemm.flydsl import (
    f8f8bf16_rowwise_preshuffle,
    f8f8f16_rowwise_preshuffle,
)


def _ref(xq, wq, x_scale, w_scale, dtype):
    a = xq.to(torch.float32) * x_scale.view(-1, 1)
    b = wq.to(torch.float32) * w_scale.view(-1, 1)
    return torch.mm(a, b.T).to(dtype)


# (M, N, K)
SHAPES = [
    (16, 5120, 8192),
    (32, 1024, 2048),
    (128, 1024, 2048),
    (512, 5120, 8192),
    (1024, 2048, 8192),
]


@unittest.skipUnless(torch.cuda.is_available(), "requires ROCm-compatible device")
class FlyDSLPreshuffleTest(unittest.TestCase):
    def _run(self, op, torch_dtype, M, N, K):
        torch.manual_seed(0)
        dev = "cuda"
        x = torch.randn(M, K, device=dev, dtype=torch.float32) * 0.1
        w = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.1
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)

        # CK op contract: caller pre-shuffles the weights.
        wq_shuffled = ck_preshuffle(wq, 16)

        out = op(xq, wq_shuffled, x_scale, w_scale)
        ref = _ref(xq, wq, x_scale.flatten(), w_scale.flatten(), torch_dtype)

        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch_dtype)
        torch.testing.assert_close(
            out.to(torch.float32), ref.to(torch.float32), rtol=0.1, atol=0.1
        )

    # ── Direct Python-function path ──────────────────────────────────────────
    def test_bf16(self):
        for (M, N, K) in SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run(f8f8bf16_rowwise_preshuffle, torch.bfloat16, M, N, K)

    def test_fp16(self):
        for (M, N, K) in SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run(f8f8f16_rowwise_preshuffle, torch.float16, M, N, K)

    # ── Registered-op path (the WP-G3 "Replaces" requirement) ────────────────
    def test_registered_ops_exist(self):
        self.assertTrue(hasattr(torch.ops.mslk, "f8f8bf16_rowwise_preshuffle"))
        self.assertTrue(hasattr(torch.ops.mslk, "f8f8f16_rowwise_preshuffle"))

    def test_op_bf16(self):
        for (M, N, K) in SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run(
                    torch.ops.mslk.f8f8bf16_rowwise_preshuffle, torch.bfloat16, M, N, K
                )

    def test_op_fp16(self):
        for (M, N, K) in SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run(
                    torch.ops.mslk.f8f8f16_rowwise_preshuffle, torch.float16, M, N, K
                )

    def test_op_fp16_dtype_not_miswired(self):
        # Guards against the historic bug where f8f8f16_* was wired to the bf16
        # implementation (csrc/gemm/gemm_ops.cpp). fp16 op must return fp16.
        M, N, K = 128, 1024, 2048
        x = torch.randn(M, K, device="cuda") * 0.1
        w = torch.randn(N, K, device="cuda") * 0.1
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        out = torch.ops.mslk.f8f8f16_rowwise_preshuffle(
            xq, ck_preshuffle(wq, 16), x_scale, w_scale
        )
        self.assertEqual(out.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
