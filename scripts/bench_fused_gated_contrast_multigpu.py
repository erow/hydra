#!/usr/bin/env python3
"""Multi-GPU large-batch latency/memory bench for fused gated contrast.

Mode: independent data-parallel replicas (one process per GPU). Each GPU runs
the fused op locally — no DDP all_gather in the timed path (matches measuring
the op itself after keys are already gathered).

Shapes (MoCo defaults: dim K=256; N = gathered keys):
  - Primary stress: B=4096, N=4096, K=256/512  (each GPU full large batch)
  - MoCo-like 4-GPU global-B=4096: B_local=1024, N=4096, K=256/512
  - Related: B=4096, N=16384 (4× gather of B=4096), K=256

Example:
  srun --overlap --jobid=JOB -w NODE bash -lc '
    export PYTHONPATH=.../SimLAP/src
    .../python3 .../bench_fused_gated_contrast_multigpu.py --gpus 4 --out ...'
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moco.fused_gated_contrast import (  # noqa: E402
    fused_gated_contrast,
    reference_gated_contrast,
    reference_gated_contrast_no_materialize,
)


def _sync(device: torch.device):
    torch.cuda.synchronize(device)


def _latency_ms(fn, device, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "iters": iters,
    }


def _peak_mem_gb(fn, device) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    try:
        fn()
        _sync(device)
        oom = False
        err = None
    except torch.cuda.OutOfMemoryError as e:
        oom = True
        err = str(e)
        torch.cuda.empty_cache()
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
        "allocated_after_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "oom": oom,
        "oom_error": err,
    }


def bench_shape_on_device(
    device: torch.device,
    B: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    skip_ref: bool,
) -> dict:
    torch.manual_seed(0 + device.index)
    x1 = torch.randn(B, K, device=device, dtype=dtype)
    x2 = torch.randn(N, K, device=device, dtype=dtype)
    gate = torch.rand(B, K, device=device, dtype=dtype).clamp(0.05, 1.0)

    # Autotune outside timed region.
    try:
        _ = fused_gated_contrast(x1, x2, gate, backend="triton")
        _sync(device)
    except Exception as e:
        return {
            "B": B,
            "N": N,
            "K": K,
            "dtype": str(dtype).replace("torch.", ""),
            "error": f"triton warmup failed: {e}",
        }

    backends = {}
    # torch (no materialize)
    backends["torch"] = {
        "latency": _latency_ms(
            lambda: reference_gated_contrast_no_materialize(x1, x2, gate),
            device,
            warmup,
            iters,
        ),
        "memory": _peak_mem_gb(
            lambda: reference_gated_contrast_no_materialize(x1, x2, gate),
            device,
        ),
    }
    # triton
    backends["triton"] = {
        "latency": _latency_ms(
            lambda: fused_gated_contrast(x1, x2, gate, backend="triton"),
            device,
            warmup,
            iters,
        ),
        "memory": _peak_mem_gb(
            lambda: fused_gated_contrast(x1, x2, gate, backend="triton"),
            device,
        ),
    }
    # reference (may OOM)
    if not skip_ref:
        try:
            # dry-run to catch OOM before latency loop
            torch.cuda.empty_cache()
            _ = reference_gated_contrast(x1, x2, gate)
            _sync(device)
            backends["reference"] = {
                "latency": _latency_ms(
                    lambda: reference_gated_contrast(x1, x2, gate),
                    device,
                    warmup,
                    max(5, iters // 2),  # fewer iters if huge
                ),
                "memory": _peak_mem_gb(
                    lambda: reference_gated_contrast(x1, x2, gate),
                    device,
                ),
            }
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            backends["reference"] = {
                "latency": None,
                "memory": {
                    "peak_allocated_gb": None,
                    "peak_reserved_gb": None,
                    "oom": True,
                    "oom_error": str(e),
                },
            }

    # Free inputs before next shape
    del x1, x2, gate
    torch.cuda.empty_cache()

    return {
        "B": B,
        "N": N,
        "K": K,
        "dtype": str(dtype).replace("torch.", ""),
        "backends": backends,
        "approx_ref_intermediate_gb": (B * N * K * dtype.itemsize) / (1024**3),
    }


def worker(rank: int, world: int, args_dict: dict, return_dict):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args_dict["dtype"]]

    shapes = [tuple(s) for s in args_dict["shapes"]]
    rows = []
    for B, N, K in shapes:
        print(
            f"[rank{rank}] bench B={B} N={N} K={K} dtype={args_dict['dtype']} "
            f"on {torch.cuda.get_device_name(device)}",
            flush=True,
        )
        row = bench_shape_on_device(
            device,
            B,
            N,
            K,
            dtype,
            warmup=args_dict["warmup"],
            iters=args_dict["iters"],
            skip_ref=args_dict["skip_ref"],
        )
        row["gpu_index"] = rank
        row["gpu_name"] = torch.cuda.get_device_name(device)
        rows.append(row)
        # compact console line
        for name, b in row.get("backends", {}).items():
            lat = b.get("latency")
            mem = b.get("memory") or {}
            if lat is None:
                print(
                    f"[rank{rank}] {name}: OOM  mem_oom={mem.get('oom')}",
                    flush=True,
                )
            else:
                print(
                    f"[rank{rank}] {name}: median={lat['median_ms']:.3f}ms "
                    f"mean={lat['mean_ms']:.3f}ms "
                    f"peak_alloc={mem.get('peak_allocated_gb', float('nan')):.3f}GB "
                    f"peak_rsv={mem.get('peak_reserved_gb', float('nan')):.3f}GB",
                    flush=True,
                )

    return_dict[rank] = {
        "rank": rank,
        "gpu_index": rank,
        "gpu_name": torch.cuda.get_device_name(device),
        "total_mem_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--skip-ref", action="store_true")
    parser.add_argument(
        "--shapes",
        default="",
        help="Comma list B:N:K (default: large-bs MoCo set)",
    )
    parser.add_argument("--out", default="", help="JSON report path (scratch)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    n_visible = torch.cuda.device_count()
    n_gpus = n_visible if args.gpus <= 0 else min(args.gpus, n_visible)
    if n_gpus < 1:
        raise SystemExit("No GPUs visible")

    if args.shapes:
        shapes = []
        for part in args.shapes.split(","):
            b, n, k = part.strip().split(":")
            shapes.append((int(b), int(n), int(k)))
    else:
        # Primary: each GPU B=4096; MoCo-like local 1024 with gathered N=4096;
        # optional full gather of B=4096 across 4 GPUs → N=16384.
        shapes = [
            (4096, 4096, 256),
            (4096, 4096, 512),
            (1024, 4096, 256),
            (1024, 4096, 512),
            (4096, 16384, 256),
        ]

    print(
        f"visible_gpus={n_visible} using={n_gpus} dtype={args.dtype} "
        f"shapes={shapes} mode=independent_replicas (no all_gather)",
        flush=True,
    )

    args_dict = {
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "skip_ref": args.skip_ref,
        "shapes": shapes,
    }

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    procs = []
    for rank in range(n_gpus):
        p = mp.Process(target=worker, args=(rank, n_gpus, args_dict, return_dict))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: worker exitcode={p.exitcode}", flush=True)

    per_gpu = [return_dict[r] for r in range(n_gpus) if r in return_dict]
    # Aggregate: mean across GPUs of median latency / peak mem for each shape+backend
    agg = []
    if per_gpu:
        n_shapes = len(per_gpu[0]["results"])
        for si in range(n_shapes):
            shape_row = {"B": None, "N": None, "K": None, "dtype": args.dtype, "backends": {}}
            for backend in ("reference", "torch", "triton"):
                meds, means, allocs, rsvs, ooms = [], [], [], [], []
                for g in per_gpu:
                    r = g["results"][si]
                    shape_row["B"], shape_row["N"], shape_row["K"] = r["B"], r["N"], r["K"]
                    b = r.get("backends", {}).get(backend)
                    if not b:
                        continue
                    if b.get("latency") is None or (b.get("memory") or {}).get("oom"):
                        ooms.append(g["gpu_index"])
                        continue
                    meds.append(b["latency"]["median_ms"])
                    means.append(b["latency"]["mean_ms"])
                    allocs.append(b["memory"]["peak_allocated_gb"])
                    rsvs.append(b["memory"]["peak_reserved_gb"])
                shape_row["backends"][backend] = {
                    "median_ms_mean_over_gpus": statistics.mean(meds) if meds else None,
                    "mean_ms_mean_over_gpus": statistics.mean(means) if means else None,
                    "peak_allocated_gb_mean": statistics.mean(allocs) if allocs else None,
                    "peak_reserved_gb_mean": statistics.mean(rsvs) if rsvs else None,
                    "peak_allocated_gb_max": max(allocs) if allocs else None,
                    "oom_gpus": ooms,
                    "n_ok": len(meds),
                }
            agg.append(shape_row)

    report = {
        "mode": "independent_data_parallel_replicas",
        "note": (
            "Each of N GPUs runs the fused op locally with given (B,N,K). "
            "No DDP all_gather in timed path. MoCo training gathers keys first "
            "(concat_all_gather) so N≈B_local*world; primary stress uses B=4096 "
            "per GPU."
        ),
        "n_gpus": n_gpus,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "per_gpu": per_gpu,
        "aggregate": agg,
    }

    # Markdown tables
    print("\n=== Aggregate (mean over GPUs) ===")
    print(
        "| GPUs | B | N | K | dtype | backend | median_ms | mean_ms | "
        "peak_alloc_GB | peak_rsv_GB | OOM |"
    )
    print("|--:|--:|--:|--:|:--|:--|---:|---:|---:|---:|:--|")
    for row in agg:
        for backend, b in row["backends"].items():
            oom = "yes" if b["oom_gpus"] else "no"
            med = b["median_ms_mean_over_gpus"]
            mean = b["mean_ms_mean_over_gpus"]
            pa = b["peak_allocated_gb_mean"]
            pr = b["peak_reserved_gb_mean"]
            print(
                f"| {n_gpus} | {row['B']} | {row['N']} | {row['K']} | {row['dtype']} | "
                f"{backend} | "
                f"{'' if med is None else f'{med:.3f}'} | "
                f"{'' if mean is None else f'{mean:.3f}'} | "
                f"{'' if pa is None else f'{pa:.3f}'} | "
                f"{'' if pr is None else f'{pr:.3f}'} | {oom} |"
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
