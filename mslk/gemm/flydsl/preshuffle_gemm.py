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


# Cache compiled JitFunctions keyed by (M, N, K, tiles, out_dtype).
_COMPILE_CACHE: dict = {}


# Pick (tile_m, tile_n, tile_k) for a shape (placeholder until Phase B autotuning).
def _select_tiles(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """Heuristic tile selection satisfying the FlyDSL kernel's constraints.

    Constraints (fp8, elem_bytes=1, 256 threads, 16B vector loads):
      * tile_k % 64 == 0
      * (tile_m * tile_k) % 4096 == 0   (bytes_per_thread_a % 16 == 0)
      * (tile_n * tile_k) % 4096 == 0
      * K % tile_k == 0 and N % tile_n == 0
    Phase B will replace this with an autotuned shape->config table.
    """
    if M <= 16:
        tile_m, tk_choices = 16, (512, 256)
    elif M <= 32:
        tile_m, tk_choices = 32, (256, 128)
    elif M <= 64:
        tile_m, tk_choices = 64, (256, 128)
    else:
        tile_m, tk_choices = 128, (128,)

    tile_k = next((tk for tk in tk_choices if K % tk == 0 and (tile_m * tk) % 4096 == 0), None)
    if tile_k is None:
        for tk in (512, 256, 128, 64):
            if K % tk == 0 and (tile_m * tk) % 4096 == 0:
                tile_k = tk
                break
    if tile_k is None:
        raise ValueError(f"No valid tile_k for K={K} (must be a multiple of 64 dividing K).")

    tile_n = next(
        (tn for tn in (256, 128, 64) if N % tn == 0 and (tn * tile_k) % 4096 == 0),
        None,
    )
    if tile_n is None:
        raise ValueError(f"No valid tile_n for N={N}, tile_k={tile_k}.")

    return tile_m, tile_n, tile_k


# Compile (or reuse a cached) FlyDSL JitFunction for this shape/dtype/tiles.
def _get_launch_fn(M, N, K, out_dtype, tiles):
    key = (M, N, K, out_dtype, tiles)
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        compile_preshuffle_gemm_a8, _arch = _kernel_api()
        tile_m, tile_n, tile_k = tiles
        fn = compile_preshuffle_gemm_a8(
            M=M, N=N, K=K,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype="fp8", out_dtype=out_dtype, lds_stage=2,
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

    tiles = _select_tiles(M, N, K)
    launch_fn = _get_launch_fn(M, N, K, out_dtype, tiles)

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
