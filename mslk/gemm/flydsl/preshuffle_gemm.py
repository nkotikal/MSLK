# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlyDSL-backed FP8 rowwise preshuffle GEMM.

Phase A (enablement) of the WP-G3 migration: wrap the FlyDSL
``kernels/preshuffle_gemm.py`` kernel behind the existing MSLK op names
``f8f8bf16_rowwise_preshuffle`` / ``f8f8f16_rowwise_preshuffle`` so it is a
drop-in replacement for the CK (``DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle``)
implementation.

Contract (identical to the CK op, see csrc/gemm/gemm_ops.cpp and
bench/gemm/gemm_ops.py):

    out = f8f8{bf16,f16}_rowwise_preshuffle(XQ, WQ, x_scale, w_scale,
                                            bias=None, use_fast_accum=True)

    XQ        : (M, K)   float8_e4m3 (fnuz on gfx942, fn on gfx950)
    WQ        : (N, K)   float8, ALREADY preshuffled by mslk.quantize.shuffle.ck_preshuffle(w, 16)
    x_scale   : (M,)     float32, per-row scale of A
    w_scale   : (N,)     float32, per-row scale of B
    bias      : unsupported on AMD (must be None)
    returns   : (M, N)   bf16 or f16

The FlyDSL in-kernel B layout has been verified byte-identical to CK's
ck_preshuffle (host shuffle round-trips; GEMM matches the torch dequant
reference for both bf16 and fp16 out on gfx942).
"""

import functools
from typing import Optional, Tuple

import torch


# Import the FlyDSL kernel builder + arch probe once, only when first needed.
# FlyDSL is a runtime dependency of MSLK; the kernel template ships as the
# `kernels.preshuffle_gemm` module (see FlyDSL packaging).
#
# REBASE POINT: the FlyDSL kernel namespace is not finalized. When the FlyDSL
# integration PR lands in MSLK, update this single import to match wherever the
# kernels end up (e.g. `flydsl.kernels.preshuffle_gemm` or
# `mslk.gemm.flydsl.kernels.preshuffle_gemm`). This is the only line that needs
# to change.
@functools.lru_cache(maxsize=1)
def _kernel_api():
    from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
    from flydsl.runtime.device import get_rocm_arch

    return compile_preshuffle_gemm_a8, str(get_rocm_arch())


# Cache compiled JitFunctions keyed by (M, N, K, out_dtype, config).
_COMPILE_CACHE: dict = {}


# Kernel config: (tile_m, tile_n, tile_k, lds_stage, use_async_copy[, waves_per_eu]).
# The optional 6th element (waves_per_eu occupancy hint) is emitted by the
# autotuner's --wide sweep; 5-tuples (waves_per_eu=None) remain valid.
Config = Tuple[int, ...]

# ─────────────────────────────────────────────────────────────────────────────
# Autotuned shape -> best config table.
#
# Source: bench/gemm/autotune_preshuffle.py --model all --wide --ck-profiler,
# gfx942 (fp8_e4m3fnuz). Full sweep = 156 shapes (13 M x 12 (N,K)); see
# autotune_preshuffle_LOG.md and autotune_results_full.json.
#   key   = (M, N, K) GEMM problem shape
#   value = (tile_m, tile_n, tile_k, lds_stage, use_async_copy, waves_per_eu)
#
# POLICY: only shapes where the tuned FlyDSL config is >= upstream CK
# (ckProfiler, all instances) are baked here (65 of 154 benchmarked). Shapes
# where CK is still faster are intentionally OMITTED so they fall through to
# _fallback_config instead of shipping a known-slower config; they are
# catalogued in _CK_FASTER below as the remaining Phase-B work. Trailing comment
# on each row is the measured FlyDSL-vs-CK speedup.
# ─────────────────────────────────────────────────────────────────────────────
_TUNED: dict = {
    # ── decode (M <= 128): FlyDSL's strong regime, up to 2.5x ──
    (1, 896, 5120): (16, 64, 512, 1, True, None),  # 2.48x
    (1, 1280, 8192): (16, 64, 512, 1, True, None),  # 2.31x
    (1, 2048, 5120): (16, 64, 512, 1, False, 4),  # 2.13x
    (1, 5120, 1024): (16, 64, 512, 1, True, 2),  # 1.52x
    (1, 7168, 8192): (16, 64, 256, 2, False, None),  # 1.30x
    (1, 8192, 1024): (16, 64, 256, 1, True, None),  # 1.39x
    (1, 8192, 3584): (16, 64, 512, 1, True, None),  # 1.14x
    (16, 896, 5120): (16, 64, 512, 1, False, None),  # 2.36x
    (16, 1280, 8192): (16, 64, 512, 1, True, 4),  # 2.54x
    (16, 2048, 5120): (16, 64, 512, 1, True, 2),  # 2.29x
    (16, 5120, 1024): (16, 64, 512, 1, False, None),  # 1.41x
    (16, 7168, 8192): (16, 64, 512, 1, True, 4),  # 1.26x
    (16, 8192, 1024): (16, 64, 512, 1, True, 4),  # 1.33x
    (16, 8192, 3584): (16, 64, 512, 1, False, 4),  # 1.05x
    (16, 13312, 6656): (16, 64, 256, 2, False, None),  # 1.12x
    (32, 896, 5120): (16, 64, 512, 1, True, 4),  # 2.41x
    (32, 1280, 8192): (16, 64, 512, 1, True, 2),  # 2.51x
    (32, 2048, 5120): (16, 64, 512, 1, False, 2),  # 2.17x
    (32, 5120, 1024): (32, 64, 512, 1, True, None),  # 1.29x
    (32, 7168, 8192): (32, 64, 512, 2, False, 2),  # 1.17x
    (32, 8192, 1024): (32, 128, 256, 1, False, 2),  # 1.10x
    (32, 8192, 3584): (32, 64, 512, 2, False, 2),  # 1.05x
    (32, 13312, 6656): (32, 64, 512, 2, False, None),  # 1.09x
    (64, 896, 5120): (16, 64, 512, 1, True, 2),  # 2.33x
    (64, 1280, 8192): (16, 64, 512, 1, True, None),  # 1.92x
    (64, 2048, 5120): (32, 64, 512, 1, False, None),  # 1.71x
    (64, 5120, 640): (32, 64, 128, 2, True, 2),  # 1.00x
    (64, 5120, 1024): (64, 64, 256, 1, False, None),  # 1.14x
    (64, 7168, 8192): (64, 64, 512, 1, True, 2),  # 1.05x
    (96, 896, 5120): (16, 64, 512, 1, False, 2),  # 2.10x
    (96, 1280, 8192): (32, 64, 512, 1, False, None),  # 1.71x
    (96, 2048, 5120): (32, 64, 512, 1, False, 2),  # 1.56x
    (96, 5120, 1024): (64, 64, 256, 1, False, None),  # 1.04x
    (128, 896, 5120): (32, 64, 512, 1, True, None),  # 1.81x
    (128, 1280, 8192): (32, 64, 512, 2, False, None),  # 1.59x
    (128, 2048, 5120): (32, 64, 512, 1, False, 2),  # 1.35x
    # ── prefill (M > 128): only the shapes where FlyDSL still edges CK ──
    (256, 896, 5120): (16, 64, 512, 1, False, 2),  # 1.62x
    (256, 1280, 8192): (32, 64, 512, 1, False, 2),  # 1.26x
    (256, 2048, 5120): (64, 64, 512, 1, True, None),  # 1.11x
    (512, 896, 5120): (32, 64, 512, 1, False, None),  # 1.47x
    (512, 1280, 8192): (64, 64, 512, 1, False, None),  # 1.02x
    (1024, 896, 5120): (32, 128, 512, 2, False, 4),  # 1.05x
    (2048, 8192, 1024): (128, 256, 128, 1, False, None),  # 1.01x
    (4096, 7168, 8192): (128, 256, 128, 1, False, 2),  # 1.03x
    (4096, 8192, 3584): (128, 128, 128, 1, True, None),  # 1.06x
    (4096, 13312, 16384): (128, 256, 128, 1, False, None),  # 1.03x
    (4096, 16384, 6656): (128, 256, 128, 1, True, None),  # 1.01x
    (4096, 16384, 16384): (128, 256, 128, 1, True, 2),  # 1.02x
    (8192, 2048, 5120): (128, 256, 128, 1, False, None),  # 1.00x
    (8192, 5120, 1024): (128, 128, 128, 1, False, 2),  # 1.01x
    (8192, 7168, 8192): (128, 256, 128, 1, False, 2),  # 1.00x
    (8192, 8192, 1024): (128, 256, 128, 1, False, None),  # 1.09x
    (8192, 8192, 3584): (128, 256, 128, 1, False, None),  # 1.02x
    (8192, 13312, 6656): (128, 256, 128, 1, True, 2),  # 1.04x
    (8192, 13312, 16384): (128, 256, 128, 1, False, None),  # 1.05x
    (8192, 16384, 6656): (128, 256, 128, 1, False, 2),  # 1.02x
    (8192, 16384, 16384): (128, 256, 128, 1, False, 2),  # 1.04x
    (16384, 896, 5120): (128, 128, 128, 2, False, 4),  # 1.00x
    (16384, 2048, 5120): (128, 256, 128, 1, False, None),  # 1.04x
    (16384, 5120, 640): (128, 256, 128, 1, False, 2),  # 1.10x
    (16384, 5120, 1024): (128, 256, 128, 1, True, None),  # 1.13x
    (16384, 8192, 1024): (128, 256, 128, 1, True, None),  # 1.09x
    (16384, 8192, 3584): (128, 256, 128, 1, True, None),  # 1.03x
    (16384, 13312, 6656): (128, 256, 128, 1, False, 2),  # 1.04x
    (16384, 16384, 6656): (128, 256, 128, 1, False, 2),  # 1.03x
}

# ─────────────────────────────────────────────────────────────────────────────
# GAP CATALOG — shapes where upstream CK is still faster (speedup < 1.0x).
#
# These are intentionally NOT in _TUNED (they use _fallback_config at runtime).
# They are the remaining Phase-B work: the mid-prefill band (M=256..2048) and
# the large-N/K llama3_405b shapes, where CK's big-tile + split-K/K-batch
# instances win. Closing them needs GRID/KERNEL extensions (split-K, larger
# tile_n, harvest CK's winning tile from ckProfiler) — not just re-tuning the
# current grid.
#
# Value = (best_config, flydsl_us, ck_us, speedup) where
#   best_config = (tile_m, tile_n, tile_k, lds_stage, use_async_copy, waves_per_eu)
#     — the fastest FlyDSL config we DID find for the shape (kept so it can be
#       revisited/re-tested even though it lost; it's a starting point, not a win)
#   speedup = ck_us / flydsl_us < 1.0 IS the slowdown: 0.68 => FlyDSL takes
#       ~1.47x as long as CK (~47% slower); 0.99 => near-parity. Lower = worse.
# Rows grouped so the worst band (mid-prefill) is visible. 89 shapes; see
# autotune_results_full.json.
# ─────────────────────────────────────────────────────────────────────────────
_CK_FASTER: dict = {
    # decode / small-M losses (mostly llama3_405b large N,K)
    (1, 13312, 6656): ((16, 128, 256, 2, False, 4), 24.44, 24.28, 0.99),
    (1, 13312, 16384): ((16, 128, 256, 2, True, None), 53.93, 52.01, 0.96),
    (1, 16384, 6656): ((16, 64, 256, 2, False, None), 28.88, 24.92, 0.86),
    (1, 16384, 16384): ((16, 64, 256, 2, False, None), 72.97, 56.37, 0.77),
    (16, 13312, 16384): ((16, 64, 256, 2, False, None), 56.28, 54.09, 0.96),
    (16, 16384, 6656): ((16, 64, 256, 2, False, None), 29.54, 26.06, 0.88),
    (16, 16384, 16384): ((16, 64, 512, 1, True, None), 80.29, 61.58, 0.77),
    (32, 5120, 640): ((32, 64, 128, 2, True, 2), 5.04, 4.86, 0.96),
    (32, 13312, 16384): ((32, 128, 512, 2, False, None), 61.77, 54.03, 0.87),
    (32, 16384, 6656): ((32, 64, 512, 2, False, 4), 33.17, 27.36, 0.82),
    (32, 16384, 16384): ((32, 64, 512, 1, True, 2), 83.02, 63.34, 0.76),
    (64, 8192, 1024): ((64, 64, 256, 1, True, None), 6.19, 6.03, 0.97),
    (64, 8192, 3584): ((64, 64, 512, 2, False, 2), 14.53, 13.68, 0.94),
    (64, 13312, 6656): ((64, 64, 512, 2, False, None), 33.16, 30.5, 0.92),
    (64, 13312, 16384): ((64, 128, 512, 1, False, 4), 70.76, 54.88, 0.78),
    (64, 16384, 6656): ((64, 64, 512, 2, False, 2), 38.88, 27.58, 0.71),
    (64, 16384, 16384): ((64, 64, 512, 1, False, 2), 88.3, 66.13, 0.75),
    (96, 5120, 640): ((64, 64, 128, 1, True, 2), 5.32, 5.0, 0.94),
    (96, 7168, 8192): ((64, 64, 512, 2, False, 4), 36.13, 28.56, 0.79),
    (96, 8192, 1024): ((64, 64, 256, 1, True, 4), 6.94, 6.25, 0.9),
    (96, 8192, 3584): ((64, 64, 512, 2, False, 2), 18.92, 13.34, 0.7),
    (96, 13312, 6656): ((128, 128, 256, 1, True, 4), 44.3, 31.04, 0.7),
    (96, 13312, 16384): ((128, 64, 256, 1, True, 4), 97.18, 72.12, 0.74),
    (96, 16384, 6656): ((128, 128, 256, 1, True, 4), 49.96, 38.36, 0.77),
    (96, 16384, 16384): ((128, 64, 256, 1, True, None), 116.45, 83.16, 0.71),
    (128, 5120, 640): ((64, 64, 128, 1, True, None), 5.44, 5.38, 0.99),
    (128, 5120, 1024): ((64, 64, 256, 1, True, None), 6.24, 6.2, 0.99),
    (128, 7168, 8192): ((64, 64, 512, 1, False, 4), 37.58, 28.9, 0.77),
    (128, 8192, 1024): ((64, 64, 256, 1, True, 2), 6.93, 6.33, 0.91),
    (128, 8192, 3584): ((64, 64, 512, 2, False, None), 19.86, 13.48, 0.68),
    (128, 13312, 6656): ((64, 128, 256, 2, False, 4), 47.18, 34.66, 0.73),
    (128, 13312, 16384): ((128, 64, 256, 2, True, 2), 106.77, 72.8, 0.68),
    (128, 16384, 6656): ((128, 128, 256, 1, False, None), 53.63, 39.47, 0.74),
    (128, 16384, 16384): ((128, 64, 256, 2, True, 4), 119.1, 91.28, 0.77),
    # mid-prefill band (M=256..2048) — the core remaining gap
    (256, 5120, 640): ((64, 64, 128, 1, True, None), 6.51, 5.87, 0.9),
    (256, 5120, 1024): ((128, 64, 256, 1, True, 2), 8.61, 6.54, 0.76),
    (256, 7168, 8192): ((128, 64, 256, 2, True, 4), 57.48, 40.23, 0.7),
    (256, 8192, 1024): ((128, 64, 256, 1, True, 2), 9.71, 8.15, 0.84),
    (256, 8192, 3584): ((128, 64, 256, 1, False, 4), 30.25, 21.54, 0.71),
    (256, 13312, 6656): ((128, 128, 256, 1, False, None), 67.32, 54.94, 0.82),
    (256, 13312, 16384): ((64, 256, 128, 2, False, None), 174.96, 123.26, 0.7),
    (256, 16384, 6656): ((128, 128, 256, 2, False, 2), 80.48, 63.22, 0.79),
    (256, 16384, 16384): ((128, 128, 128, 2, False, None), 201.05, 149.59, 0.74),
    (512, 2048, 5120): ((32, 128, 256, 2, False, 2), 23.54, 18.17, 0.77),
    (512, 5120, 640): ((128, 128, 128, 2, False, 4), 8.72, 8.0, 0.92),
    (512, 5120, 1024): ((128, 128, 128, 1, False, 2), 12.59, 10.22, 0.81),
    (512, 7168, 8192): ((128, 128, 256, 2, False, 2), 85.87, 67.37, 0.78),
    (512, 8192, 1024): ((128, 128, 256, 1, True, 2), 16.73, 13.66, 0.82),
    (512, 8192, 3584): ((128, 128, 256, 1, True, 4), 44.17, 36.58, 0.83),
    (512, 13312, 6656): ((128, 256, 128, 1, False, None), 108.77, 86.99, 0.8),
    (512, 13312, 16384): ((128, 256, 512, 1, False, None), 258.48, 208.44, 0.81),
    (512, 16384, 6656): ((128, 256, 128, 2, False, 4), 123.67, 105.85, 0.86),
    (512, 16384, 16384): ((128, 256, 512, 1, False, None), 292.21, 277.62, 0.95),
    (1024, 1280, 8192): ((64, 128, 512, 1, True, 2), 36.3, 30.85, 0.85),
    (1024, 2048, 5120): ((64, 128, 512, 1, False, 4), 32.3, 29.27, 0.91),
    (1024, 5120, 640): ((64, 128, 128, 2, False, None), 13.96, 12.76, 0.91),
    (1024, 5120, 1024): ((64, 128, 128, 2, False, None), 21.16, 17.27, 0.82),
    (1024, 7168, 8192): ((64, 128, 128, 2, False, None), 132.56, 110.13, 0.83),
    (1024, 8192, 1024): ((128, 256, 128, 2, False, None), 28.73, 24.25, 0.84),
    (1024, 8192, 3584): ((128, 256, 128, 2, False, 4), 76.27, 65.4, 0.86),
    (1024, 13312, 6656): ((128, 128, 128, 1, False, 2), 183.78, 155.41, 0.85),
    (1024, 13312, 16384): ((64, 256, 256, 2, False, 4), 470.6, 410.78, 0.87),
    (1024, 16384, 6656): ((128, 256, 128, 1, False, 2), 234.74, 214.75, 0.91),
    (1024, 16384, 16384): ((128, 256, 128, 1, True, 2), 574.5, 541.85, 0.94),
    (2048, 896, 5120): ((64, 128, 512, 1, True, 2), 27.73, 26.59, 0.96),
    (2048, 1280, 8192): ((64, 256, 256, 1, False, 2), 58.73, 52.84, 0.9),
    (2048, 2048, 5120): ((64, 256, 256, 1, False, None), 52.71, 50.5, 0.96),
    (2048, 5120, 640): ((64, 128, 128, 1, True, None), 23.86, 21.41, 0.9),
    (2048, 5120, 1024): ((64, 256, 128, 1, True, 2), 35.65, 29.56, 0.83),
    (2048, 7168, 8192): ((64, 256, 128, 1, True, None), 223.3, 205.87, 0.92),
    (2048, 8192, 3584): ((128, 256, 128, 1, True, None), 119.68, 118.03, 0.99),
    (2048, 13312, 6656): ((128, 256, 128, 1, True, None), 326.6, 320.2, 0.98),
    (2048, 13312, 16384): ((128, 256, 128, 1, False, 2), 791.36, 769.18, 0.97),
    (2048, 16384, 6656): ((64, 256, 256, 2, False, 4), 426.18, 402.14, 0.94),
    (2048, 16384, 16384): ((128, 256, 128, 1, True, 2), 998.93, 976.21, 0.98),
    # large prefill that narrowly loses (near parity)
    (4096, 896, 5120): ((64, 128, 256, 2, False, None), 48.94, 46.55, 0.95),
    (4096, 1280, 8192): ((128, 256, 256, 2, False, 2), 96.65, 90.43, 0.94),
    (4096, 2048, 5120): ((128, 256, 128, 1, False, 2), 88.55, 87.38, 0.99),
    (4096, 5120, 640): ((64, 256, 128, 2, False, 2), 41.05, 38.24, 0.93),
    (4096, 5120, 1024): ((64, 256, 128, 1, True, 2), 59.08, 53.7, 0.91),
    (4096, 8192, 1024): ((128, 128, 128, 1, False, 2), 85.87, 79.73, 0.93),
    (4096, 13312, 6656): ((64, 256, 256, 2, False, 2), 638.59, 633.0, 0.99),
    (8192, 896, 5120): ((64, 128, 128, 2, False, 4), 85.47, 79.86, 0.93),
    (8192, 1280, 8192): ((64, 128, 256, 1, True, 4), 182.02, 157.78, 0.87),
    (8192, 5120, 640): ((64, 256, 128, 2, False, None), 73.87, 71.36, 0.97),
    (16384, 1280, 8192): ((128, 256, 128, 1, True, None), 343.86, 319.41, 0.93),
    (16384, 7168, 8192): ((128, 256, 128, 1, False, None), 1572.05, 1550.13, 0.99),
    (16384, 13312, 16384): ((128, 256, 128, 1, False, 2), 5935.3, 5858.45, 0.99),
    (16384, 16384, 16384): ((128, 256, 128, 1, False, 2), 7320.2, 7156.85, 0.98),
}

# Shapes the autotuner's grid skipped (K=640 at M<=16: tile_m is pinned to 16,
# and 16*tile_k is never 4096-aligned for any tile_k that divides 640). These
# are NOT dead at runtime — _fallback_config pads tile_m up to 32 (32*128=4096),
# which is legal and correct; they just aren't autotuned. Kept for the record.
_UNTUNED_PADDED = {
    (1, 5120, 640),
    (16, 5120, 640),
}


def _fallback_config(M: int, N: int, K: int) -> Config:
    """Constraint-safe config for shapes not in the tuned table.

    Defaults follow the pattern the autotuner consistently found best for this op:
    smallest legal tile_n (these GEMMs are skinny), largest legal tile_k,
    tile_m tracking M, lds_stage=1, no async, default occupancy.

    tile_m search: we prefer a tile_m close to M, but fall through to a *larger*
    tile_m when the smaller ones cannot satisfy the fp8 load-alignment rule for
    this K. Example: N=5120, K=640 at M<=16 — tile_m=16 has no legal tile_k
    (K=640 only divides tile_k=128, and 16*128 is not 4096-aligned), but
    tile_m=32 works (32*128=4096); the kernel pads the extra rows at runtime.
    This is what closes the former "no valid config" corner.

    Constraints (fp8): tile_k%64==0, (tile_m*tile_k)%4096==0,
    (tile_n*tile_k)%4096==0, K%tile_k==0, N%tile_n==0.
    """
    target = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    # Try tile_m at/above the target first (minimal padding), largest tile_k,
    # smallest tile_n — return the first fully-legal combination.
    for tile_m in [t for t in (16, 32, 64, 128, 256) if t >= target]:
        tile_k = next((tk for tk in (512, 256, 128, 64)
                       if K % tk == 0 and (tile_m * tk) % 4096 == 0), None)
        if tile_k is None:
            continue
        tile_n = next((tn for tn in (64, 128, 256)
                       if N % tn == 0 and (tn * tile_k) % 4096 == 0), None)
        if tile_n is None:
            continue
        return (tile_m, tile_n, tile_k, 1, False, None)
    raise ValueError(
        f"No fp8-legal preshuffle config for (M={M}, N={N}, K={K}); "
        f"K must have a divisor tile_k in (64,128,256,512) with a matching "
        f"4096-aligned tile_m/tile_n."
    )


# Pick the kernel config for a shape: tuned table if present, else fallback.
def _select_config(M: int, N: int, K: int) -> Config:
    cfg = _TUNED.get((M, N, K))
    return cfg if cfg is not None else _fallback_config(M, N, K)


# Compile (or reuse a cached) FlyDSL JitFunction for this shape/dtype/config.
def _get_launch_fn(M, N, K, out_dtype, cfg: Config):
    key = (M, N, K, out_dtype, cfg)
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        compile_preshuffle_gemm_a8, _arch = _kernel_api()
        # Accept 5-tuple (legacy) or 6-tuple (waves_per_eu from --wide tuning).
        tile_m, tile_n, tile_k, lds_stage, use_async_copy, *rest = cfg
        waves_per_eu = rest[0] if rest else None
        fn = compile_preshuffle_gemm_a8(
            M=M, N=N, K=K,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype="fp8", out_dtype=out_dtype,
            lds_stage=lds_stage, use_async_copy=use_async_copy,
            waves_per_eu=waves_per_eu,
        )
        _COMPILE_CACHE[key] = fn
    return fn


# Reinterpret an fp8 tensor as raw int8 bytes for passing to the kernel.
def _as_i8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


# Core adapter: marshal MSLK inputs, run the FlyDSL kernel, return (M, N) output.
def _rowwise_preshuffle_impl(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: str,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    if bias is not None:
        raise NotImplementedError("AMD preshuffle GEMM does not support fused bias.")
    if not use_fast_accum:
        raise NotImplementedError("AMD does not support disabling use_fast_accum.")
    if out_dtype not in ("bf16", "fp16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'fp16', got {out_dtype!r}")

    assert XQ.is_cuda and WQ.is_cuda, "inputs must be on device"
    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    # XQ may carry leading dims (…, K); flatten to (M, K).
    *lead, K = XQ.shape
    M = 1
    for d in lead:
        M *= d
    XQ2 = XQ.reshape(M, K)
    N = WQ.shape[0]
    assert WQ.shape[1] == K, f"WQ K-dim {WQ.shape[1]} != XQ K-dim {K}"

    if M == 0 or N == 0 or K == 0:
        out = XQ.new_zeros((*lead, N), dtype=torch_out_dtype)
        return out

    cfg = _select_config(M, N, K)
    launch_fn = _get_launch_fn(M, N, K, out_dtype, cfg)

    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=XQ.device)
    launch_fn(
        c_out.view(-1),
        _as_i8(XQ2.contiguous().view(-1)),
        _as_i8(WQ.contiguous().view(-1)),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        M, N, torch.cuda.current_stream(),
    )
    return c_out.reshape(*lead, N)


# Public op: FP8 rowwise preshuffle GEMM with bf16 output.
def f8f8bf16_rowwise_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """FP8 rowwise preshuffle GEMM, bf16 output (FlyDSL backend)."""
    return _rowwise_preshuffle_impl(
        XQ, WQ, x_scale, w_scale, "bf16", bias=bias, use_fast_accum=use_fast_accum
    )


# Public op: FP8 rowwise preshuffle GEMM with fp16 output.
def f8f8f16_rowwise_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """FP8 rowwise preshuffle GEMM, fp16 output (FlyDSL backend)."""
    return _rowwise_preshuffle_impl(
        XQ, WQ, x_scale, w_scale, "fp16", bias=bias, use_fast_accum=use_fast_accum
    )
