#!/usr/bin/env python3
"""Correctness + latency benchmark for fused gated contrast on GPU.

Compares:
  - ungated:      plain MoCo contrast, no gate (MoCo.contrastive_loss logits)
  - reference:    Filter.forward + Filter.contrast (materializes [B,N,K])
  - torch_einsum: original einsum fusion without [B,N,K]
  - torch:        two cuBLAS GEMMs + fp32 epilogue, no [B,N,K]
  - triton:       fused double-GEMM Triton kernel (tensor cores)
  - triton_reduce: non-tensor-core per-row Triton fallback

``ungated`` is the cost floor: it is what the same batch would cost without any
gating, so ``gated / ungated`` is the price of the per-sample subspace.

Example (Isambard allocation):
  srun --overlap --jobid=5915103 -w nid010060 -n1 \\
    env PYTHONPATH=/lus/lfs1aip2/projects/u6gd/jiantao/SimLAP/src \\
    /lus/lfs1aip2/projects/u6gd/jiantao/FastSSL/.venv/bin/python3 \\
    /lus/lfs1aip2/projects/u6gd/jiantao/SimLAP/src/moco/bench_fused_gated_contrast.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Allow running as a script without installing the package.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moco.fused_gated_contrast import (  # noqa: E402
    check_correctness,
    fused_gated_contrast,
    reference_gated_contrast,
    torch_gated_contrast,
    torch_gated_contrast_einsum,
)

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def ungated_contrast(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Plain contrast without a gate, as written in ``MoCo.contrastive_loss``.

    Normalize both sides once, then one [B,K] x [K,N] GEMM. This is the cost
    floor of the objective: the gate is what turns the shared key matrix into a
    per-query one.
    """
    q = F.normalize(x1, dim=1)
    k = F.normalize(x2, dim=1)
    return torch.einsum("nc,mc->nm", q, k)


def _backends(include_ref: bool) -> dict:
    fns = {
        "ungated": lambda x1, x2, g: ungated_contrast(x1, x2),
        "torch_einsum": lambda x1, x2, g: torch_gated_contrast_einsum(x1, x2, g),
        "torch": lambda x1, x2, g: torch_gated_contrast(x1, x2, g),
        "triton": lambda x1, x2, g: fused_gated_contrast(
            x1, x2, g, backend="triton", method="gemm"
        ),
        "triton_reduce": lambda x1, x2, g: fused_gated_contrast(
            x1, x2, g, backend="triton", method="reduce"
        ),
    }
    if include_ref:
        fns = {"reference": reference_gated_contrast, **fns}
    return fns


def _time_ms(fn, warmup: int, iters: int) -> dict:
    """CUDA-event timing.

    ``median_ms`` synchronizes every call, so it includes the CPU launch cost of
    the op in isolation. ``pipelined_ms`` times a batch of back-to-back calls,
    letting the CPU run ahead, which is what a training step that is not
    CPU-bound actually pays. ``cpu_ms`` is the enqueue cost alone.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    batch = max(10, min(50, iters))
    pipelined = []
    for _ in range(3):
        start.record()
        for _ in range(batch):
            fn()
        end.record()
        end.synchronize()
        pipelined.append(start.elapsed_time(end) / batch)

    t0 = time.perf_counter()
    for _ in range(batch):
        fn()
    cpu_ms = (time.perf_counter() - t0) * 1000.0 / batch
    torch.cuda.synchronize()

    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "pipelined_ms": statistics.median(pipelined),
        "cpu_ms": cpu_ms,
    }


def _peak_mem_mib(fn, trials: int = 3) -> float:
    """Extra device memory the call needs on top of what is already live.

    Repeated because the caching allocator can satisfy an early call from
    blocks that were already counted as live, which under-reports the peak.
    """
    worst = 0.0
    for _ in range(trials):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del out
        worst = max(worst, (peak - before) / (1024**2))
    return worst


def _graph_ms(fn, warmup: int = 5) -> Optional[float]:
    """GPU-only time via CUDA graph replay (no CPU launch cost at all)."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = []
        for _ in range(3):
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            best.append(start.elapsed_time(end) / 20)
        del graph
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return statistics.median(best)
    except Exception as exc:  # noqa: BLE001
        print(f"    [graph] capture failed: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return None


def run_correctness(device: str, shapes, dtypes) -> list[dict]:
    rows = []
    for dtype in dtypes:
        for B, N, K in shapes:
            ok, stats = check_correctness(B=B, N=N, K=K, dtype=dtype, device=device)
            rows.append(stats)
            print(
                f"[correctness] {'PASS' if ok else 'FAIL'} "
                f"dtype={stats['dtype']:8s} B,N,K={stats['shape']} "
                f"torch={stats['torch_vs_ref_max']:.3e} "
                f"triton={stats.get('triton_vs_ref_max', float('nan')):.3e} "
                f"triton_reduce={stats.get('triton_reduce_vs_ref_max', float('nan')):.3e} "
                f"grad(x1,x2,g)=("
                f"{stats.get('grad_x1_rel_max', float('nan')):.2e},"
                f"{stats.get('grad_x2_rel_max', float('nan')):.2e},"
                f"{stats.get('grad_gate_rel_max', float('nan')):.2e})",
                flush=True,
            )
    return rows


def run_benchmark(
    device: str,
    dtype: torch.dtype,
    shapes,
    warmup: int,
    iters: int,
    skip_ref: bool,
    bwd: bool,
) -> list[dict]:
    rows = []
    for B, N, K in shapes:
        torch.manual_seed(0)
        x1 = torch.randn(B, K, device=device, dtype=dtype)
        x2 = torch.randn(N, K, device=device, dtype=dtype)
        gate = torch.rand(B, K, device=device, dtype=dtype).clamp(0.05, 1.0)
        torch.cuda.synchronize()

        # Reference materializes [B,N,K]; skip when it clearly cannot fit.
        ref_bytes = B * N * K * torch.finfo(dtype).bits // 8
        free_bytes = torch.cuda.mem_get_info()[0]
        include_ref = (not skip_ref) and (3 * ref_bytes < free_bytes)

        row = {
            "B": B,
            "N": N,
            "K": K,
            "dtype": str(dtype).replace("torch.", ""),
            "forward": {},
            "backward": {},
        }

        for name, fn in _backends(include_ref).items():
            call = lambda fn=fn: fn(x1, x2, gate)  # noqa: E731
            try:
                call()  # warm autotune / cuBLAS heuristics outside timing
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row["forward"][name] = {"oom": True}
                continue
            entry = _time_ms(call, warmup, iters)
            entry["peak_mem_mib"] = _peak_mem_mib(call)
            entry["graph_ms"] = _graph_ms(call)
            entry["oom"] = False
            row["forward"][name] = entry

        if bwd:
            grad_out = torch.randn(B, N, device=device, dtype=dtype)
            bwd_fns = {
                "ungated": lambda a, b, c: ungated_contrast(a, b),
                "torch": lambda a, b, c: torch_gated_contrast(a, b, c),
                "triton": lambda a, b, c: fused_gated_contrast(
                    a, b, c, backend="triton"
                ),
            }
            if include_ref:
                bwd_fns = {"reference": reference_gated_contrast, **bwd_fns}
            for name, fn in bwd_fns.items():
                a = x1.clone().requires_grad_(True)
                b = x2.clone().requires_grad_(True)
                c = gate.clone().requires_grad_(True)

                def step(fn=fn, a=a, b=b, c=c):
                    a.grad = b.grad = c.grad = None
                    out = fn(a, b, c)
                    out.backward(grad_out)

                try:
                    step()
                    torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row["backward"][name] = {"oom": True}
                    continue
                entry = _time_ms(step, max(2, warmup // 2), max(5, iters // 2))
                entry["peak_mem_mib"] = _peak_mem_mib(lambda step=step: step())
                entry["oom"] = False
                row["backward"][name] = entry
            del grad_out

        fwd = row["forward"]
        t_tri = fwd.get("triton", {}).get("median_ms")
        t_torch = fwd.get("torch", {}).get("median_ms")
        t_te = fwd.get("torch_einsum", {}).get("median_ms")
        t_ref = fwd.get("reference", {}).get("median_ms")
        t_un = fwd.get("ungated", {}).get("median_ms")
        row["speedup_vs_torch"] = (
            round(t_torch / t_tri, 3) if t_tri and t_torch else None
        )
        # Cost of gating relative to the ungated objective (lower is better).
        row["gate_overhead_triton"] = round(t_tri / t_un, 3) if t_tri and t_un else None
        row["gate_overhead_torch"] = (
            round(t_torch / t_un, 3) if t_torch and t_un else None
        )
        row["gate_overhead_ref"] = round(t_ref / t_un, 3) if t_ref and t_un else None
        row["speedup_vs_torch_einsum"] = round(t_te / t_tri, 3) if t_tri and t_te else None
        row["speedup_vs_ref"] = round(t_ref / t_tri, 3) if t_tri and t_ref else None

        p_tri = fwd.get("triton", {}).get("pipelined_ms")
        p_torch = fwd.get("torch", {}).get("pipelined_ms")
        p_te = fwd.get("torch_einsum", {}).get("pipelined_ms")
        row["pipelined_speedup_vs_torch"] = (
            round(p_torch / p_tri, 3) if p_tri and p_torch else None
        )
        row["pipelined_speedup_vs_torch_einsum"] = (
            round(p_te / p_tri, 3) if p_tri and p_te else None
        )
        gr_tri = fwd.get("triton", {}).get("graph_ms")
        gr_torch = fwd.get("torch", {}).get("graph_ms")
        gr_te = fwd.get("torch_einsum", {}).get("graph_ms")
        row["graph_speedup_vs_torch"] = (
            round(gr_torch / gr_tri, 3) if gr_tri and gr_torch else None
        )
        row["graph_speedup_vs_torch_einsum"] = (
            round(gr_te / gr_tri, 3) if gr_tri and gr_te else None
        )

        def _f(v):
            return "   n/a" if v is None else f"{v:7.3f}"

        print(
            f"[bench] B={B:5d} N={N:5d} K={K:4d}  ungated={_f(t_un)}  "
            f"ref={_f(t_ref)}  "
            f"einsum={_f(t_te)}  torch={_f(t_torch)}  triton={_f(t_tri)}  "
            f"reduce={_f(fwd.get('triton_reduce', {}).get('median_ms'))}  "
            f"| triton vs torch = {row['speedup_vs_torch']}x  "
            f"| gate cost vs ungated: triton={row['gate_overhead_triton']}x "
            f"torch={row['gate_overhead_torch']}x ref={row['gate_overhead_ref']}x",
            flush=True,
        )
        print(
            f"        pipelined: einsum={_f(p_te)}  torch={_f(p_torch)}  "
            f"triton={_f(p_tri)}  | {row['pipelined_speedup_vs_torch']}x   "
            f"cpu: torch={_f(fwd.get('torch', {}).get('cpu_ms'))} "
            f"triton={_f(fwd.get('triton', {}).get('cpu_ms'))}",
            flush=True,
        )
        print(
            f"        gpu-only (cuda graph): ref={_f(fwd.get('reference', {}).get('graph_ms'))}  "
            f"einsum={_f(gr_te)}  torch={_f(gr_torch)}  triton={_f(gr_tri)}  "
            f"| {row['graph_speedup_vs_torch']}x",
            flush=True,
        )
        if row["backward"]:
            print(
                "        bwd: "
                + "  ".join(
                    f"{k}={_f(v.get('median_ms'))}" for k, v in row["backward"].items()
                ),
                flush=True,
            )
        rows.append(row)

        del x1, x2, gate
        torch.cuda.empty_cache()
    return rows


def _parse_shapes(text: str):
    shapes = []
    for part in text.split(","):
        b, n, k = part.strip().split(":")
        shapes.append((int(b), int(n), int(k)))
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=list(_DTYPES))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default="", help="JSON output path (prefer scratch)")
    parser.add_argument("--shapes", default="", help="Comma list B:N:K")
    parser.add_argument(
        "--preset",
        default="default",
        choices=("default", "bs-sweep"),
        help="Shape set when --shapes is not given",
    )
    parser.add_argument("--correctness-shapes", default="")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-ref", action="store_true")
    parser.add_argument("--skip-bwd", action="store_true")
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 for fp32 matmuls (both cuBLAS and the Triton dot)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = args.device
    dtype = _DTYPES[args.dtype]
    print(
        f"device={device} name={torch.cuda.get_device_name(0)} "
        f"dtype={args.dtype} torch={torch.__version__} "
        f"tf32={torch.backends.cuda.matmul.allow_tf32}",
        flush=True,
    )

    # Square sweep: N = B, i.e. contrast against the batch itself (single GPU,
    # or per-rank after an all-gather of the same total size).
    _BS_SWEEP = [
        (b, b, k) for k in (256, 512) for b in (256, 512, 1024, 2048, 4096)
    ]
    # Typical MoCo/SimLAP: local B, gathered N (=B*world), dim K in {128,256,512}.
    _DEFAULT = [
        (64, 256, 128),
        (128, 512, 256),
        (256, 1024, 256),
        (512, 2048, 256),
        (512, 2048, 512),
        (1024, 4096, 256),
        (1024, 4096, 512),
        (4096, 4096, 256),
        (4096, 4096, 512),
        (4096, 16384, 256),
    ]
    if args.shapes:
        shapes = _parse_shapes(args.shapes)
    elif args.preset == "bs-sweep":
        shapes = _BS_SWEEP
    else:
        shapes = _DEFAULT
    c_shapes = (
        _parse_shapes(args.correctness_shapes)
        if args.correctness_shapes
        else [(32, 128, 256), (128, 512, 256), (256, 1024, 512), (1024, 1024, 256)]
    )

    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "correctness": [],
        "benchmark": [],
    }

    if not args.skip_correctness:
        print("=== correctness ===", flush=True)
        report["correctness"] = run_correctness(
            device, c_shapes, (torch.float32, torch.float16, torch.bfloat16)
        )

    if not args.skip_bench:
        print("=== benchmark ===", flush=True)
        report["benchmark"] = run_benchmark(
            device,
            dtype,
            shapes,
            warmup=args.warmup,
            iters=args.iters,
            skip_ref=args.skip_ref,
            bwd=not args.skip_bwd,
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}", flush=True)

    if report["benchmark"]:
        print(
            "\n| B | N | K | ungated ms | ref ms | einsum ms | torch ms | "
            "triton ms | triton/torch | gpu torch | gpu triton | gpu speedup | "
            "triton MiB | torch MiB | ref MiB |"
        )
        print("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for r in report["benchmark"]:
            f = r["forward"]

            def g(name, field="median_ms"):
                v = f.get(name, {})
                if v.get("oom"):
                    return "OOM"
                v = v.get(field)
                return "" if v is None else f"{v:.3f}"

            print(
                f"| {r['B']} | {r['N']} | {r['K']} | {g('ungated')} | "
                f"{g('reference')} | "
                f"{g('torch_einsum')} | {g('torch')} | {g('triton')} | "
                f"{r['speedup_vs_torch']}x | {g('torch','graph_ms')} | "
                f"{g('triton','graph_ms')} | "
                f"{r['graph_speedup_vs_torch']}x | "
                f"{g('triton','peak_mem_mib')} | {g('torch','peak_mem_mib')} | "
                f"{g('reference','peak_mem_mib')} |"
            )

        print("\n=== gate cost vs ungated contrast (median ms / peak MiB) ===")
        print(
            "| B | N | K | ungated ms | ungated MiB | ref ms | ref MiB | "
            "torch ms | torch MiB | triton ms | triton MiB | triton/ungated |"
        )
        print("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for r in report["benchmark"]:
            f = r["forward"]

            def g(name, field="median_ms"):
                v = f.get(name, {})
                if v.get("oom"):
                    return "OOM"
                v = v.get(field)
                return "n/a" if v is None else f"{v:.3f}"

            print(
                f"| {r['B']} | {r['N']} | {r['K']} | {g('ungated')} | "
                f"{g('ungated','peak_mem_mib')} | {g('reference')} | "
                f"{g('reference','peak_mem_mib')} | {g('torch')} | "
                f"{g('torch','peak_mem_mib')} | {g('triton')} | "
                f"{g('triton','peak_mem_mib')} | {r['gate_overhead_triton']}x |"
            )

    failed = [c for c in report["correctness"] if not c.get("ok")]
    if failed:
        raise SystemExit(f"correctness failures: {len(failed)}")


if __name__ == "__main__":
    main()
