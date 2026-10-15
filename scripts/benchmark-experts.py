"""Compare isolated expert DLL builds using real Q4_K model weights.

Uses the runtime job lease, never unloads another process, and never replaces
installed libraries. CUDA device interop is explicitly enabled for that case.
Numbers are expert-layer microbenchmarks, not end-to-end synthesis timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fish_drama3.hybrid import ExpertBridge
from fish_drama3.watchdog import guarded_job_lock


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trace", action="store_true", help="Rotate 25 CPU layers with fresh top-8 routes per frame")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--baseline-cache", action="store_true")
    parser.add_argument("--baseline-poll", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["baseline", "candidate", "device"],
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be 1..100")
    if any(not 1 <= value <= 128 for value in args.tokens):
        parser.error("tokens must be 1..128")
    if any(not 1 <= value <= 256 for value in args.threads):
        parser.error("threads must be 1..256")
    if not 1 <= args.frames <= 100 or (args.trace and args.device != "cpu"):
        parser.error("trace requires CPU device and 1..100 frames")
    if not 0 <= args.baseline_poll <= 100:
        parser.error("baseline poll must be 0..100")
    return args


def inputs(tokens: int, device: str) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1969 + tokens)
    values = torch.randn((tokens, 2048), generator=generator) * 0.02
    ids = torch.stack([torch.randperm(256, generator=generator)[:8] for _ in range(tokens)])
    mixing = torch.rand((tokens, 8), generator=generator)
    mixing /= mixing.sum(dim=-1, keepdim=True)
    return values.to(device), ids.to(device), mixing.to(device)


def measure(bridge: ExpertBridge, layer: int, tensors: tuple[torch.Tensor, ...],
            repeats: int) -> tuple[dict[str, Any], torch.Tensor]:
    for _ in range(2):
        bridge.forward(layer, *tensors)
    bridge.profile_summary(reset=True)
    samples = []
    for _ in range(repeats):
        if tensors[0].is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = bridge.forward(layer, *tensors)
        if result.is_cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    idle_started, idle_cpu = time.perf_counter(), time.process_time()
    time.sleep(0.1)
    idle_cpu = time.process_time() - idle_cpu
    idle_wall = time.perf_counter() - idle_started
    return dict(median_seconds=statistics.median(samples),
                minimum_seconds=min(samples), samples_seconds=samples,
                idle_cpu_seconds=idle_cpu, idle_wall_seconds=idle_wall,
                profile=bridge.profile_summary()), result.cpu()


def run_case(args: argparse.Namespace, library: Path, label: str,
             device_bridge: bool) -> tuple[list[dict], dict[tuple, torch.Tensor]]:
    os.environ["FISH_DRAMA3_GPU_DEVICE_BRIDGE"] = str(int(device_bridge))
    gpu = args.device == "cuda"
    bridge = ExpertBridge(args.model, threads=args.threads[0], gpu_layers=int(gpu),
                          library=library.resolve(), profile=True)
    records, outputs = [], {}
    try:
        bridge.configure_performance(poll=args.baseline_poll if label == "baseline" else 0,
                                     cache=label != "baseline" or args.baseline_cache)
        for threads in args.threads:
            bridge.threads = threads
            if args.trace:
                record, output = measure_trace(bridge, args.frames)
                record.update(build=label, tokens=1, threads=threads, device_bridge=False)
                records.append(record)
                outputs[(threads, 1)] = output
                print(json.dumps(record), flush=True)
                continue
            for tokens in args.tokens:
                tensors = inputs(tokens, args.device)
                tensors = (tensors[0].to(getattr(torch, args.dtype)), *tensors[1:])
                record, output = measure(bridge, 39 if gpu else 4, tensors, args.repeats)
                record.update(build=label, tokens=tokens, threads=threads,
                              device_bridge=device_bridge)
                records.append(record)
                outputs[(threads, tokens)] = output
                print(json.dumps(record), flush=True)
    finally:
        bridge.close()
        if gpu:
            torch.cuda.empty_cache()
    return records, outputs


def measure_trace(bridge: ExpertBridge, frames: int) -> tuple[dict, torch.Tensor]:
    """Warm the entire fixed routing trace so baseline sees no disk-page penalty."""
    generator = torch.Generator().manual_seed(1971)
    trace = []
    for _ in range(frames):
        frame = []
        for layer in range(25):
            values = torch.randn((1, 2048), generator=generator) * 0.02
            ids = torch.randperm(256, generator=generator)[:8].reshape(1, 8)
            weights = torch.rand((1, 8), generator=generator)
            frame.append((layer, (values, ids, weights / weights.sum())))
        trace.append(frame)
    for frame in trace:
        for layer, tensors in frame:
            bridge.forward(layer, *tensors)
    bridge.profile_summary(reset=True)
    samples, outputs = [], []
    for frame in trace:
        started = time.perf_counter()
        for layer, tensors in frame:
            outputs.append(bridge.forward(layer, *tensors))
        samples.append(time.perf_counter() - started)
    idle_started, idle_cpu = time.perf_counter(), time.process_time()
    time.sleep(0.1)
    record = dict(median_seconds=statistics.median(samples), samples_seconds=samples,
                  trace_layers=25, frames=frames, idle_cpu_seconds=time.process_time() - idle_cpu,
                  idle_wall_seconds=time.perf_counter() - idle_started,
                  profile=bridge.profile_summary())
    return record, torch.stack(outputs)


def main() -> None:
    args = options()
    torch.set_num_threads(1)
    if args.worker:
        library = args.baseline if args.worker == "baseline" else args.candidate
        records, outputs = run_case(args, library, args.worker, args.worker == "device")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records), encoding="utf-8")
        np.savez(args.output.with_suffix(".npz"),
                 **{f"{key[0]}:{key[1]}": value.numpy() for key, value in outputs.items()})
        return
    with guarded_job_lock():
        baseline, reference = isolated_case(args, "baseline")
        candidate, outputs = isolated_case(args, "candidate")
        cases = baseline + candidate
        variants = [("candidate", outputs)]
        if args.device == "cuda":
            records, device_outputs = isolated_case(args, "device")
            cases.extend(records)
            variants.append(("device", device_outputs))
        parity = []
        for name, variant in variants:
            for key, actual in variant.items():
                expected = reference[key]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                parity.append(dict(build=name, threads=key[0], tokens=key[1],
                                   maximum_absolute_error=float((actual - expected).abs().max())))
        report = dict(device=args.device, dtype=args.dtype, model=str(args.model.resolve()),
                      torch=torch.__version__, cases=cases, parity=parity,
                      hardware=dict(processor=platform.processor(), logical_cpus=os.cpu_count(),
                                    gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None),
                      omp_wait_policy=os.environ.get("OMP_WAIT_POLICY"),
                      libraries={name: library_identity(path) for name, path in
                                 (("baseline", args.baseline), ("candidate", args.candidate))},
                      note="Layer microbenchmark; not complete synthesis throughput")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def isolated_case(args: argparse.Namespace, label: str) -> tuple[list[dict], dict]:
    """A fresh process prevents Windows DLL-name reuse from mixing CPU backends."""
    output = args.output.with_name(f"{args.output.stem}-{label}.worker.json")
    command = [sys.executable, str(Path(__file__).resolve()),
               "--model", str(args.model), "--baseline", str(args.baseline),
               "--candidate", str(args.candidate), "--device", args.device,
               "--dtype", args.dtype, "--repeats", str(args.repeats),
               "--threads", *map(str, args.threads), "--tokens", *map(str, args.tokens),
               "--frames", str(args.frames), "--output", str(output), "--worker", label]
    command.extend(["--baseline-poll", str(args.baseline_poll)])
    if args.baseline_cache:
        command.append("--baseline-cache")
    if args.trace:
        command.append("--trace")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    records = json.loads(output.read_text(encoding="utf-8"))
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as arrays:
        outputs = {tuple(map(int, key.split(":"))): torch.from_numpy(arrays[key]) for key in arrays}
    print(f"{label}: {len(records)} isolated benchmark cases completed", flush=True)
    return records, outputs


def library_identity(path: Path) -> dict:
    backend = path.parent / "ggml-cpu.dll"
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cpu_backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
