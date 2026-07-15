#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Phase B autotuner for the FlyDSL FP8 rowwise-preshuffle GEMM (WP-G3).

Sweeps a constraint-generated config grid over a target shape set, keeps the
numerically-correct configs, times them, and records the fastest per shape. The
winners become the shape->config table baked into
``mslk/gemm/flydsl/preshuffle_gemm.py::_select_config``.

Crash isolation (step 5): some configs (large tile_k + deep preload) overflow
LDS and *hard-abort* the process at compile time (uncatchable). Each shape is
therefore tuned in a child process that streams per-config results; if it dies,
the driver resumes that shape from the config after the crasher.

Runs standalone on ROCm; no compiled ``mslk.so`` required.

Usage:
    PYTHONPATH=<mslk-root> python bench/gemm/autotune_preshuffle.py \
        [--model llama3_70b|llama4|llama3_405b|all] [--limit N] [--out results.json]
"""

import argparse
import json
import os
import subprocess
import sys

import torch


# ── Make FlyDSL importable (installed dep preferred; dev-checkout fallback) ────
def _ensure_flydsl_on_path() -> None:
    try:
        import kernels  # noqa: F401  (FlyDSL repo-level package)
        return
    except ImportError:
        pass
    root = os.environ.get("FLYDSL_ROOT")
    if root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.abspath(os.path.join(here, "..", "..", "..", "flyDSL"))
        if os.path.isdir(cand):
            root = cand
    if root and root not in sys.path:
        sys.path.insert(0, root)


_ensure_flydsl_on_path()

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

ARCH = str(get_rocm_arch())
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

TRIAL_TIMEOUT_S = 300

# ── Shape sets (from bench/gemm/gemm_bench.py registry) ───────────────────────
SHAPE_SETS = {
    "llama3_70b": [(M, N, K) for M in (1, 16, 32, 64, 96, 128)
                   for (N, K) in ((1280, 8192), (8192, 1024), (7168, 8192), (8192, 3584))],
    "llama4": [(M, N, K) for M in (1, 16, 32, 64, 96, 128)
               for (N, K) in ((896, 5120), (5120, 640), (2048, 5120), (5120, 1024))],
    "llama3_405b": [(M, N, K) for M in (1, 16, 32, 64, 96, 128)
                    for (N, K) in ((13312, 6656), (13312, 16384), (16384, 6656), (16384, 16384))],
}


# ── Step 3: constraint-based config grid ──────────────────────────────────────
def _tile_m_choices(M):
    nextpow2 = 1 << max(0, (M - 1)).bit_length()
    cap = max(16, nextpow2)
    return [t for t in (16, 32, 64, 128) if t <= cap]


def gen_configs(M, N, K):
    """All valid (tile_m,tile_n,tile_k,lds_stage,use_async_copy) for this shape.

    fp8 (elem_bytes=1), 256 threads, 16B vector loads:
      tile_k % 64 == 0, (tile_m*tile_k) % 4096 == 0, (tile_n*tile_k) % 4096 == 0,
      N % tile_n == 0, K % tile_k == 0.
    """
    cfgs = []
    for tm in _tile_m_choices(M):
        for tk in (128, 256, 512):
            if K % tk or (tm * tk) % 4096:
                continue
            for tn in (64, 128, 256):
                if N % tn or (tn * tk) % 4096:
                    continue
                for lds in (2, 1):
                    for aco in (False, True):
                        cfgs.append((tm, tn, tk, lds, aco))
    return cfgs


def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def _make_problem(M, N, K):
    from tests.utils import pertoken_quant, shuffle_weight
    torch.manual_seed(0)
    a = torch.rand(M, K, device="cuda", dtype=torch.float32)
    b = torch.rand(N, K, device="cuda", dtype=torch.float32)
    a_q, sa = pertoken_quant(a, quant_dtype=DTYPE_FP8)
    b_q, sb = pertoken_quant(b, quant_dtype=DTYPE_FP8)
    b_shuf = shuffle_weight(b_q.contiguous(), layout=(16, 16))
    c_ref = (a_q.float() * sa.view(-1, 1)) @ (b_q.float() * sb.view(-1, 1)).T
    return a_q.contiguous(), b_shuf, sa.view(-1).contiguous(), sb.view(-1).contiguous(), c_ref


def _time_one(M, N, K, cfg, prob):
    from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
    from tests.test_common import run_perftest
    tm, tn, tk, lds, aco = cfg
    a_q, b_shuf, sa, sb, c_ref = prob
    fn = compile_preshuffle_gemm_a8(M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
                                    in_dtype="fp8", out_dtype="bf16",
                                    lds_stage=lds, use_async_copy=aco)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

    def launch(c, a, b, x, w):
        fn(c.view(-1), _as_i8(a.view(-1)), _as_i8(b.view(-1)), x, w,
           M, N, torch.cuda.current_stream())

    launch(c, a_q, b_shuf, sa, sb)
    torch.cuda.synchronize()
    if not torch.allclose(c.float(), c_ref, rtol=0.1, atol=0.1):
        return None
    _, us = run_perftest(launch, c, a_q, b_shuf, sa, sb, num_iters=20, num_warmup=5)
    return float(us)


# ── Step 5: per-shape worker (streams "R <idx> <us|INCORRECT>", then "DONE") ───
def _worker(argv):
    M, N, K, start = int(argv[0]), int(argv[1]), int(argv[2]), int(argv[3])
    prob = _make_problem(M, N, K)
    cfgs = gen_configs(M, N, K)
    for i in range(start, len(cfgs)):
        us = _time_one(M, N, K, cfgs[i], prob)  # may hard-abort here (LDS overflow)
        print(f"R {i} {us if us is not None else 'INCORRECT'}", flush=True)
    print("DONE", flush=True)
    return 0


# ── Step 8: aiter baseline (same B-preshuffle as CK) ──────────────────────────
def _baseline_worker(argv):
    import aiter
    from tests.test_common import run_perftest
    M, N, K = int(argv[0]), int(argv[1]), int(argv[2])
    a_q, b_shuf, sa, sb, _ = _make_problem(M, N, K)

    def launch(a, b, x, w):
        return aiter.gemm_a8w8_bpreshuffle(a, b, x, w, None, torch.bfloat16)

    _, us = run_perftest(launch, a_q, b_shuf, sa.view(-1, 1), sb.view(-1, 1),
                         num_iters=20, num_warmup=5)
    print(f"US={float(us)}", flush=True)
    return 0


def _run_baseline(M, N, K):
    cmd = [sys.executable, os.path.abspath(__file__), "--baseline",
           str(M), str(N), str(K)]
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TRIAL_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    for line in p.stdout.splitlines():
        if line.startswith("US="):
            return float(line[3:])
    return None


def _tune_shape(M, N, K, iters_note=""):
    """Drive one shape across child processes, resuming past any crasher."""
    cfgs = gen_configs(M, N, K)
    results = {}  # idx -> us
    crashed = []
    start = 0
    while start < len(cfgs):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               str(M), str(N), str(K), str(start)]
        env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        last = start - 1
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=TRIAL_TIMEOUT_S, env=env)
            out = p.stdout
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        done = False
        for line in out.splitlines():
            if line.startswith("R "):
                _, i, val = line.split(maxsplit=2)
                i = int(i); last = i
                if val not in ("INCORRECT", "None"):
                    results[i] = float(val)
            elif line.strip() == "DONE":
                done = True
        if done:
            break
        # process died after config `last`; the culprit is `last + 1`.
        culprit = last + 1
        if culprit < len(cfgs):
            crashed.append(cfgs[culprit])
        start = culprit + 1
    return results, crashed, cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3_70b",
                    choices=list(SHAPE_SETS) + ["all"])
    ap.add_argument("--limit", type=int, default=0, help="max shapes (0 = all)")
    ap.add_argument("--out", default="autotune_results.json")
    ap.add_argument("--baseline", action="store_true",
                    help="also benchmark aiter.gemm_a8w8_bpreshuffle (CK-equivalent)")
    args = ap.parse_args()

    if args.model == "all":
        shapes, seen = [], set()
        for s in SHAPE_SETS.values():
            for t in s:
                if t not in seen:
                    seen.add(t); shapes.append(t)
    else:
        shapes = SHAPE_SETS[args.model]
    if args.limit:
        shapes = shapes[: args.limit]

    print(f"# arch={ARCH} fp8={DTYPE_FP8} model={args.model} shapes={len(shapes)}", flush=True)
    table = {}
    for (M, N, K) in shapes:
        results, crashed, cfgs = _tune_shape(M, N, K)
        n_ok = len(results)
        if not results:
            print(f"M={M:<4} N={N:<5} K={K:<5}: no valid config "
                  f"({len(crashed)} crashed / {len(cfgs)} total)", flush=True)
            continue
        best_i = min(results, key=results.get)
        best_us = results[best_i]
        tm, tn, tk, lds, aco = cfgs[best_i]
        tflops = 2 * M * N * K / (best_us / 1e6) / 1e12
        entry = {
            "us": round(best_us, 2), "tflops": round(tflops, 1),
            "tile_m": tm, "tile_n": tn, "tile_k": tk,
            "lds_stage": lds, "use_async_copy": aco,
        }
        ck_str = ""
        if args.baseline:
            ck_us = _run_baseline(M, N, K)
            if ck_us:
                entry["ck_us"] = round(ck_us, 2)
                entry["speedup_vs_ck"] = round(ck_us / best_us, 2)
                ck_str = f"  | CK {ck_us:6.1f} us -> {ck_us / best_us:.2f}x"
            else:
                ck_str = "  | CK n/a"
        table[f"{M},{N},{K}"] = entry
        print(f"M={M:<4} N={N:<5} K={K:<5}: BEST {best_us:7.1f} us ({tflops:5.1f} TF) "
              f"tiles=({tm},{tn},{tk}) lds={lds} async={aco}  "
              f"[{n_ok} ok / {len(crashed)} crashed / {len(cfgs)} total]{ck_str}", flush=True)

    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\n# wrote {len(table)} shape->config entries to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    torch.set_default_device("cuda")
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.exit(_worker(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--baseline":
        sys.exit(_baseline_worker(sys.argv[2:]))
    sys.exit(main())
