"""Guarded matched-prompt eager versus tower-graph synthesis benchmark.

Reports semantic generation and synchronized tower time separately. Optional
codec-first loading also measures identical-code waveform decoding. Both tower
variants use identical placement, sampling, weights, prompt and seed; acoustic
rows must match exactly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fish_drama3.hybrid import build_drama3_prompt, generate_drama3, load_model_placed
import fish_drama3.hybrid as hybrid
from fish_drama3.codec_adapter import CodecBridge
from fish_drama3.tower_graphs import (
    install_tower_graphs, prepare_tower_graphs, tower_attentions,
    uninstall_tower_graphs,
)
from fish_drama3.watchdog import guarded_job_lock


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=15)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prepare-all", action="store_true")
    parser.add_argument("--expert-cpu-library", type=Path)
    parser.add_argument("--expert-cuda-library", type=Path)
    parser.add_argument("--codec-library", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--text", default=(
        "Hello there. This is a short demonstration of Fish Audio Drama Three "
        "on Windows."))
    parser.add_argument("--text-file", type=Path)
    args = parser.parse_args()
    if args.text_file:
        args.text = args.text_file.read_text(encoding="utf-8")
    if not 1 <= args.frames <= 8192 or not 0 <= args.gpu_layers <= 40:
        parser.error("frames must be 1..8192 and GPU expert layers 0..40")
    if not 1 <= args.threads <= 256:
        parser.error("threads must be 1..256")
    if bool(args.expert_cpu_library) != bool(args.expert_cuda_library):
        parser.error("supply both CPU and CUDA expert libraries")
    if bool(args.codec) != bool(args.codec_library):
        parser.error("supply both codec weights and codec library")
    return args


@contextmanager
def native_dependencies(args: argparse.Namespace) -> Iterator[CodecBridge | None]:
    """Mirror studio codec-first loading inside this benchmark process only."""
    original, codec = hybrid.ExpertBridge, None

    def factory(model: Path, threads: int = 8, gpu_layers: int = 0,
                **kwargs: Any) -> Any:
        kwargs["library"] = (args.expert_cuda_library if gpu_layers
                             else args.expert_cpu_library).resolve()
        return original(model, threads, gpu_layers, **kwargs)

    try:
        if args.codec:
            codec = CodecBridge(args.codec, args.codec_library.resolve())
        if args.expert_cpu_library:
            hybrid.ExpertBridge = factory
        yield codec
    finally:
        hybrid.ExpertBridge = original
        if codec is not None:
            codec.close()


def loaded_dependencies() -> dict[str, dict[str, str]]:
    """Record resolved Windows donor DLLs, rather than trusting wrapper paths."""
    if os.name != "nt":
        return {}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                         ctypes.c_uint32]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint32
    records = {}
    for name in ("ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll"):
        handle = kernel.GetModuleHandleW(name)
        if not handle:
            continue
        buffer = ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        path = Path(buffer.value)
        records[name] = dict(path=str(path), sha256=hashlib.sha256(
            path.read_bytes()).hexdigest())
    return records


def environment_record(args: argparse.Namespace) -> dict[str, Any]:
    return dict(omp_wait_policy=os.environ.get("OMP_WAIT_POLICY", "unset"),
                loaded_dependencies=loaded_dependencies(),
                expert_cpu_library=str(args.expert_cpu_library),
                expert_cuda_library=str(args.expert_cuda_library),
                codec_library=str(args.codec_library))


def measure_codec(codec: CodecBridge | None, frames: torch.Tensor,
                  args: argparse.Namespace) -> dict[str, Any] | None:
    if codec is None:
        return None
    started = time.perf_counter()
    first = codec.decode(frames.numpy(), args.threads)
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    repeated = codec.decode(frames.numpy(), args.threads)
    warm_seconds = time.perf_counter() - started
    if not np.allclose(first, repeated, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Repeated codec waveform changed beyond tolerance")
    path = args.output.with_suffix(".waveform.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, repeated)
    return dict(first_seconds=first_seconds, warm_seconds=warm_seconds,
                samples=len(repeated), sha256=hashlib.sha256(
                    repeated.tobytes()).hexdigest(), waveform=str(path))


class TowerTimer:
    def __init__(self, function: Callable[..., torch.Tensor]) -> None:
        self.function, self.seconds, self.calls = function, 0.0, 0
        self.first_started, self.last_finished = 0.0, 0.0
        self.example: tuple[torch.Tensor, ...] = ()

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        torch.cuda.synchronize()
        started = time.perf_counter()
        if not self.calls:
            self.first_started = started
            self.example = tuple(value.clone() for value in args)
        result = self.function(*args, **kwargs)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - started
        self.last_finished = time.perf_counter()
        self.calls += 1
        return result


def measure(model: Any, prompt: torch.Tensor, args: argparse.Namespace,
            label: str) -> tuple[dict[str, Any], torch.Tensor]:
    timer = TowerTimer(model.audio_decoder.decode_step)
    original = model.audio_decoder.decode_step
    model.audio_decoder.decode_step = timer
    try:
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        frames, reason = generate_drama3(model, prompt, max_new_tokens=args.frames,
                                         temperature=.7, top_k=30, top_p=.7)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        model.audio_decoder.decode_step = original
    record = dict(variant=label, seconds=seconds, frames=len(frames), reason=reason,
                  frames_per_second=len(frames) / seconds, tower_seconds=timer.seconds,
                  decode_seconds=timer.last_finished - timer.first_started,
                  decode_frames_per_second=len(frames) / (
                      timer.last_finished - timer.first_started),
                  tower_calls=timer.calls, acoustic_sha256=hashlib.sha256(
                      frames.numpy().tobytes()).hexdigest())
    print(json.dumps(record), flush=True)
    model.audio_decoder._tower_benchmark_example = timer.example
    return record, frames


def check_boundaries(model: Any, seed: int) -> list[dict[str, Any]]:
    """Check every bucket transition with real tower weights and saved inputs."""
    tower = model.audio_decoder
    graphs = tower._hybrid_decode_graphs
    values = list(tower._tower_benchmark_example)
    records = []
    capacity = next(tower_attentions(tower)).k_cache.shape[2]
    positions = sorted({0, capacity - 1, *(
        position for bucket in graphs.graphs for position in (bucket - 1, bucket)
        if position < capacity)})
    for position in positions:
        values[2] = torch.tensor([position], device=tower.device)
        saved = [(cache, cache[:, :, position:position + 1].clone())
                 for attention in tower_attentions(tower)
                 for cache in (attention.k_cache, attention.v_cache)]
        try:
            torch.manual_seed(seed)
            expected = graphs.original(*values, temperature=.7, top_k=30, top_p=.7)
            expected_rng = torch.cuda.get_rng_state().clone()
            for cache, row in saved:
                cache[:, :, position:position + 1].copy_(row)
            torch.manual_seed(seed)
            actual = graphs(*values, temperature=.7, top_k=30, top_p=.7)
            records.append(dict(position=position, exact_acoustic_parity=torch.equal(
                expected, actual), exact_rng_parity=torch.equal(
                    expected_rng, torch.cuda.get_rng_state())))
        finally:
            for cache, row in saved:
                cache[:, :, position:position + 1].copy_(row)
    return records


@torch.inference_mode()
def main() -> None:
    args = options()
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    os.environ["FISH_DRAMA3_TOWER_GRAPH"] = "0"
    torch.set_num_threads(args.threads)
    with guarded_job_lock(), native_dependencies(args) as codec:
        model, bridge, placement = load_model_placed(args.bundle, threads=args.threads,
            gpu_layers=args.gpu_layers, cache_capacity=10240)
        try:
            prompt = build_drama3_prompt(
                args.text, model.tokenizer, model.config, [], [])
            baseline, expected = measure(model, prompt, args, "eager")
            warm_baseline, warmed = measure(model, prompt, args, "eager_warm")
            preparation = prepare_tower_graphs(model) if args.prepare_all else None
            if (preparation and preparation["gpu_free_bytes_after"]
                    < placement["headroom_bytes"]):
                raise RuntimeError("Graph preparation consumed required GPU headroom")
            graphs = install_tower_graphs(model)
            candidate, actual = measure(model, prompt, args, "tower_graph_first")
            repeat, repeated = measure(model, prompt, args, "tower_graph_repeat")
            boundaries = check_boundaries(model, args.seed) if args.prepare_all else []
            model.audio_decoder.decode_step = graphs.original
            after, after_frames = measure(model, prompt, args, "eager_warm_after")
            model.audio_decoder.decode_step = graphs
            parity = (torch.equal(expected, actual) and torch.equal(expected, repeated)
                      and torch.equal(expected, warmed)
                      and torch.equal(expected, after_frames)
                      and all(
                          row["exact_acoustic_parity"] and row["exact_rng_parity"]
                          for row in boundaries))
            report = dict(torch=torch.__version__, placement=placement,
                **environment_record(args),
                codec=measure_codec(codec, actual, args),
                text=args.text, seed=args.seed, max_frames=args.frames,
                cases=[baseline, warm_baseline, candidate, repeat, after],
                preparation=preparation, exact_acoustic_parity=parity,
                bucket_boundary_parity=boundaries,
                capture_seconds=graphs.capture_seconds, graph_replays=graphs.replays,
                buckets=sorted(graphs.graphs),
                note="Synchronized tower profiling; codec timing reported separately")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if not parity:
                raise RuntimeError("Tower graph changed matched-seed acoustic rows")
        finally:
            uninstall_tower_graphs(model)
            bridge.close()


if __name__ == "__main__":
    main()
