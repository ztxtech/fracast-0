#!/usr/bin/env python3
"""Benchmark the shipped Fracast inference checkpoints.

The benchmark runs in the current process, so load time and resident memory
represent a direct Python deployment.  Use `--batch` for independent-series
throughput; `forecast_batch` shares one model invocation across all rows.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but no CUDA device is available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but no MPS device is available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def resident_memory_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no latency samples were collected")
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def parameter_count(model) -> int:
    return sum(
        parameter.numel()
        for parameter in model.core.parameters()
    ) + sum(parameter.numel() for parameter in model.head.parameters())


def device_memory(device: str) -> dict[str, int | None]:
    import torch

    if device == "cuda":
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
    if device == "mps":
        return {
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "reserved_bytes": int(torch.mps.driver_allocated_memory()),
        }
    return {"allocated_bytes": None, "reserved_bytes": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "weights"))
    parser.add_argument("--weights", choices=("fp32", "w8"), default="w8")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--batch, --warmup, and --iterations must be positive")
    if args.context < 8:
        raise SystemExit("--context must be at least 8")

    device = resolve_device(args.device)
    from fracast import FracastModel

    load_started = time.perf_counter()
    model = FracastModel.from_pretrained(args.model, weights=args.weights, device=device)
    load_seconds = time.perf_counter() - load_started
    rng = np.random.default_rng(args.seed)
    context = rng.normal(size=(args.batch, args.context)).astype(np.float32)
    for _ in range(args.warmup):
        model.forecast_batch(context)
    synchronize(device)
    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        model.forecast_batch(context)
        synchronize(device)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    total_seconds = sum(latencies_ms) / 1000.0
    report = {
        "model": str(args.model),
        "weights": args.weights,
        "device": device,
        "context": args.context,
        "batch": args.batch,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "parameter_count": parameter_count(model),
        "resident_rss_bytes": resident_memory_bytes(),
        "device_memory_bytes": device_memory(device),
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "mean": float(np.mean(latencies_ms)),
        },
        "throughput_series_per_second": (
            args.batch * args.iterations / total_seconds if total_seconds else 0.0
        ),
    }
    output = args.output or ROOT / "output" / "benchmark" / f"{args.weights}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
