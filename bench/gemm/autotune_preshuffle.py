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

Shape sets cover decode (M=1..128) and, via the ``*_full`` sets, prefill
(M up to 16384) — matching the full M range CK's instances serve.

Usage:
    PYTHONPATH=<flydsl-root> python bench/gemm/autotune_preshuffle.py \
        [--model llama3_70b|llama4|llama3_405b|
                 llama3_70b_full|llama4_full|llama3_405b_full|all] \
        [--limit N] [--wide] [--out results.json] \
        [--baseline | --ck-profiler /path/to/ckProfiler]

    --wide         also sweep waves_per_eu (closes the small residual regressions)
    --baseline     quick CK proxy (aiter.gemm_a8w8_bpreshuffle)
    --ck-profiler  true upstream-CK baseline (CK's own ckProfiler, all instances)
"""

import argparse
import json
import os
import re
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

TRIAL_TIMEOUT_S = 300      # per aiter/ckProfiler baseline call
KGROUP_TIMEOUT_S = 1800    # per K-group worker window (resumes past crashes)

# ── Shape sets (from bench/gemm/gemm_bench.py registry) ───────────────────────
# M axis splits into decode (small, latency-bound) and prefill (large,
# throughput-bound). CK's 65 instances cover both regimes, so "the preshuffle
# shape set" in the ticket means decode *and* prefill — not just decode.
_DECODE_M = (1, 16, 32, 64, 96, 128)
_PREFILL_M = (256, 512, 1024, 2048, 4096, 8192, 16384)

# (N, K) layer dims per model.
_MODEL_NK = {
    "llama3_70b": ((1280, 8192), (8192, 1024), (7168, 8192), (8192, 3584)),
    "llama4": ((896, 5120), (5120, 640), (2048, 5120), (5120, 1024)),
    "llama3_405b": ((13312, 6656), (13312, 16384), (16384, 6656), (16384, 16384)),
}


def _shapes(models, Ms):
    out, seen = [], set()
    for m in models:
        for (N, K) in _MODEL_NK[m]:
            for M in Ms:
                t = (M, N, K)
                if t not in seen:
                    seen.add(t)
                    out.append(t)
    return out


SHAPE_SETS = {
    # decode-only (original coverage)
    "llama3_70b": _shapes(["llama3_70b"], _DECODE_M),
    "llama4": _shapes(["llama4"], _DECODE_M),
    "llama3_405b": _shapes(["llama3_405b"], _DECODE_M),
    # decode + prefill (full M range CK covers)
    "llama3_70b_full": _shapes(["llama3_70b"], _DECODE_M + _PREFILL_M),
    "llama4_full": _shapes(["llama4"], _DECODE_M + _PREFILL_M),
    "llama3_405b_full": _shapes(["llama3_405b"], _DECODE_M + _PREFILL_M),
}


# ── Step 3: constraint-based config grid ──────────────────────────────────────
def _tile_m_choices(M):
    # tile_m ladder capped near M; 256 is the largest tile the FlyDSL preload
    # table supports, and it only pays off for prefill (M >> 128).
    nextpow2 = 1 << max(0, (M - 1)).bit_length()
    cap = max(16, min(256, nextpow2))
    return [t for t in (16, 32, 64, 128, 256) if t <= cap]


def gen_configs(M, N, K, wide=False):
    """All valid configs for this shape.

    Config tuple: (tile_m, tile_n, tile_k, lds_stage, use_async_copy, waves_per_eu).

    fp8 (elem_bytes=1), 256 threads, 16B vector loads:
      tile_k % 64 == 0, (tile_m*tile_k) % 4096 == 0, (tile_n*tile_k) % 4096 == 0,
      N % tile_n == 0, K % tile_k == 0.

    ``wide`` additionally sweeps ``waves_per_eu`` (occupancy hint). Preload depths
    (dsrd/dvmem) are auto-derived per tile by the kernel, so they are not swept
    here; ``wide`` is what the regression note means by "widen grid".
    """
    wpes = (None,) if not wide else (None, 2, 4)
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
                        for wpe in wpes:
                            cfgs.append((tm, tn, tk, lds, aco, wpe))
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


# The compiled binary depends ONLY on (K, tile_m/n/k, lds, async, waves, dtype) —
# M and N are runtime i32 kernel args (see kernels/preshuffle_gemm.py docstring:
# "Runtime parameters: M, N"). So we compile ONCE per (K, cfg) and reuse the same
# JitFunction across every (M, N) that shares that K — this is what avoids the
# ~17x redundant recompiles the naive per-(M,N,K) loop was doing.
def _compile(K, cfg, out_dtype="bf16"):
    from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
    tm, tn, tk, lds, aco, wpe = cfg
    return compile_preshuffle_gemm_a8(
        M=0, N=0, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
        in_dtype="fp8", out_dtype=out_dtype,
        lds_stage=lds, use_async_copy=aco, waves_per_eu=wpe)


def _time_shape(fn, M, N, prob):
    """Time an already-compiled kernel on one (M, N); None if numerically wrong."""
    from tests.test_common import run_perftest
    a_q, b_shuf, sa, sb, c_ref = prob
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


# ── Shape selection (identical in parent and worker so config indices line up) ─
def _selected_shapes(model, limit):
    if model == "all":
        shapes, seen = [], set()
        for s in SHAPE_SETS.values():
            for t in s:
                if t not in seen:
                    seen.add(t); shapes.append(t)
    else:
        shapes = SHAPE_SETS[model]
    return shapes[:limit] if limit else shapes


def _mns_for_K(model, limit, K):
    return [(M, N) for (M, N, Kk) in _selected_shapes(model, limit) if Kk == K]


def _kgroup(K, mns, wide):
    """Unique configs for this K + the (M,N) shapes each config applies to.

    Compilation depends only on (K, cfg), so we tune per K: one compile serves
    every (M,N) in the group. `gen_configs` already filters tile_m by M and
    tile_n by N, so unioning per-shape configs yields exactly the applicable set.
    """
    order, seen, applies = [], set(), {}
    for (M, N) in mns:
        for cfg in gen_configs(M, N, K, wide=wide):
            if cfg not in seen:
                seen.add(cfg); order.append(cfg); applies[cfg] = []
            applies[cfg].append((M, N))
    return order, [applies[c] for c in order]


# ── K-group worker: compile each config ONCE, time across all its (M,N) ────────
# Streams "R <cfg_idx> <M> <N> <us|INCORRECT>" per timing, "C <cfg_idx>" when a
# config's shapes are all done, then "DONE". A compile may hard-abort (LDS
# overflow) — the parent resumes at the next config.
def _worker(argv):
    model, limit, K, start = argv[0], int(argv[1]), int(argv[2]), int(argv[3])
    wide = bool(int(argv[4])) if len(argv) > 4 else False
    mns = _mns_for_K(model, limit, K)
    cfgs, applies = _kgroup(K, mns, wide)
    prob_cache = {}
    for i in range(start, len(cfgs)):
        fn = _compile(K, cfgs[i])  # may hard-abort here (LDS overflow)
        for (M, N) in applies[i]:
            if (M, N) not in prob_cache:
                prob_cache[(M, N)] = _make_problem(M, N, K)
            us = _time_shape(fn, M, N, prob_cache[(M, N)])
            print(f"R {i} {M} {N} {us if us is not None else 'INCORRECT'}", flush=True)
        print(f"C {i}", flush=True)
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


# ── Upstream-CK baseline: CK's own ckProfiler (sweeps ALL CK instances) ───────
# This is the *true* upstream Composable Kernel path — it benchmarks whatever CK
# version is checked out/built (including kernels newer than MSLK's frozen 65),
# not aiter and not MSLK's precompiled dispatcher. Build once:
#   cd mslk/external/composable_kernel && mkdir build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx942 \
#         -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc .. && make ckProfiler -j
# then pass --ck-profiler <build>/bin/ckProfiler.
def _run_ck_profiler(M, N, K, ck_bin):
    # ckProfiler gemm_universal_preshuffle <dtype> <layout> <verify> <init>
    #   <log> <time> M N K StrideA StrideB StrideC KBatch [warmup iters rotMB]
    # dtype=1 -> f8f8 bf16 out; layout=1 -> A[m,k]*B[n,k]=C[m,n] (MK_NK_MN);
    # strides=-1 -> defaults; KBatch=1; verify=0; init=2 (decimal); time=1.
    cmd = [ck_bin, "gemm_universal_preshuffle", "1", "1", "0", "2", "0", "1",
           str(M), str(N), str(K), "-1", "-1", "-1", "1", "5", "20", "0"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TRIAL_TIMEOUT_S)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if p.returncode != 0:
        return None
    # CK prints per-instance "Perf: <ms> ms, ..." lines, then a summary
    # "Best Perf for datatype = ... : <ms> ms, <TFlops> TFlops, ...".
    best_ms = None
    for line in p.stdout.splitlines():
        m = re.search(r"[Bb]est\s+[Pp]erf.*?:\s*([0-9.eE+-]+)\s*ms", line)
        if m:
            best_ms = float(m.group(1))
    return best_ms * 1000.0 if best_ms is not None else None  # ms -> us


def _tune_kgroup(model, limit, K, wide=False):
    """Tune every shape sharing this K in one pass, compiling each config once.

    Drives child processes with resume-past-crash: a config that hard-aborts (or
    exceeds the window) is skipped and the next config's compile continues. Each
    completed config is marked "C <idx>" so the driver knows where to resume.

    Returns best[(M,N)] = (us, cfg), plus (#configs, #crashed_or_skipped).
    """
    mns = _mns_for_K(model, limit, K)
    cfgs, _applies = _kgroup(K, mns, wide)
    best = {}  # (M,N) -> (us, cfg)
    crashed = 0
    start = 0
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    while start < len(cfgs):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               model, str(limit), str(K), str(start), str(int(wide))]
        last_done = start - 1
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=KGROUP_TIMEOUT_S, env=env)
            out = p.stdout
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        done = False
        for line in out.splitlines():
            if line.startswith("R "):
                _, i, M, N, val = line.split()
                if val not in ("INCORRECT", "None"):
                    us = float(val); key = (int(M), int(N))
                    if key not in best or us < best[key][0]:
                        best[key] = (us, cfgs[int(i)])
            elif line.startswith("C "):
                last_done = int(line.split()[1])
            elif line.strip() == "DONE":
                done = True
        if done:
            break
        crashed += 1
        start = last_done + 2  # skip the config after the last fully-completed one
    return best, len(cfgs), crashed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3_70b",
                    choices=list(SHAPE_SETS) + ["all"])
    ap.add_argument("--limit", type=int, default=0, help="max shapes (0 = all)")
    ap.add_argument("--out", default="autotune_results.json")
    ap.add_argument("--wide", action="store_true",
                    help="widen the grid (sweep waves_per_eu) to close regressions")
    ap.add_argument("--baseline", action="store_true",
                    help="quick CK proxy via aiter.gemm_a8w8_bpreshuffle")
    ap.add_argument("--ck-profiler", default=None, metavar="PATH",
                    help="true upstream-CK baseline via CK ckProfiler binary "
                         "(overrides --baseline; sweeps all CK instances)")
    args = ap.parse_args()

    shapes = _selected_shapes(args.model, args.limit)
    # Distinct K values, in first-seen order. We tune per K so each config
    # compiles once and is reused across all (M,N) that share that K.
    ks = list(dict.fromkeys(K for (_M, _N, K) in shapes))

    ck_mode = "upstream(ckProfiler)" if args.ck_profiler else \
              "aiter" if args.baseline else "none"
    print(f"# arch={ARCH} fp8={DTYPE_FP8} model={args.model} shapes={len(shapes)} "
          f"K-groups={len(ks)} wide={args.wide} ck_baseline={ck_mode}", flush=True)

    table = {}

    def _save():
        with open(args.out, "w") as f:
            json.dump(table, f, indent=2)

    for K in ks:
        mns = _mns_for_K(args.model, args.limit, K)
        best, n_cfg, n_crash = _tune_kgroup(args.model, args.limit, K, wide=args.wide)
        print(f"# K={K}: tuned {len(mns)} shapes over {n_cfg} unique configs "
              f"({n_crash} crashed/skipped)", flush=True)
        for (M, N) in mns:
            if (M, N) not in best:
                print(f"M={M:<5} N={N:<5} K={K:<5}: no valid config", flush=True)
                continue
            best_us, (tm, tn, tk, lds, aco, wpe) = best[(M, N)]
            tflops = 2 * M * N * K / (best_us / 1e6) / 1e12
            entry = {
                "us": round(best_us, 2), "tflops": round(tflops, 1),
                "tile_m": tm, "tile_n": tn, "tile_k": tk,
                "lds_stage": lds, "use_async_copy": aco, "waves_per_eu": wpe,
            }
            ck_str = ""
            ck_us = None
            if args.ck_profiler:
                ck_us = _run_ck_profiler(M, N, K, args.ck_profiler)
                ck_src = "CK(upstream)"
            elif args.baseline:
                ck_us = _run_baseline(M, N, K)
                ck_src = "CK(aiter)"
            if ck_us:
                entry["ck_us"] = round(ck_us, 2)
                entry["ck_source"] = ck_src
                entry["speedup_vs_ck"] = round(ck_us / best_us, 2)
                ck_str = f"  | {ck_src} {ck_us:7.1f} us -> {ck_us / best_us:.2f}x"
            elif ck_mode != "none":
                ck_str = "  | CK n/a"
            table[f"{M},{N},{K}"] = entry
            _save()  # incremental: a kill preserves completed shapes
            print(f"M={M:<5} N={N:<5} K={K:<5}: BEST {best_us:7.1f} us "
                  f"({tflops:6.1f} TF) tiles=({tm},{tn},{tk}) lds={lds} "
                  f"async={aco} wpe={wpe}{ck_str}", flush=True)

    _save()
    print(f"\n# wrote {len(table)} shape->config entries to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    torch.set_default_device("cuda")
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.exit(_worker(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--baseline":
        sys.exit(_baseline_worker(sys.argv[2:]))
    sys.exit(main())
