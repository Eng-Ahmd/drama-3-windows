"""Persistent, serialized synthesis worker for the local browser application."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from fish_drama3 import __version__
from fish_drama3.cli import output_claim, publish_wave, reference_audio, verify_assets
from fish_drama3.codec_adapter import CodecBridge
from fish_drama3.watchdog import Limits, guarded_job_lock, sample_memory, violation

_EVENT_LOCK = threading.Lock()


def emit(**event: Any) -> None:
    """Keep structured events distinguishable from native library log messages."""
    with _EVENT_LOCK:
        print("DRAMA3_EVENT" + json.dumps(event, allow_nan=False), flush=True)


def preflight() -> None:
    import torch

    sample = sample_memory()
    free, total = torch.cuda.mem_get_info(0)
    sample["gpu_free_bytes"] = min(sample["gpu_free_bytes"], free)
    reason = violation(sample, memory_limits(total))
    if reason:
        raise RuntimeError(f"Memory safety reserve exceeded: {reason}. Unload other models or reduce settings.")


def memory_limits(total_bytes: int) -> Limits:
    percent = float(os.environ.get("FISH_DRAMA3_WEB_HEADROOM_PERCENT", "3"))
    if not 1 <= percent <= 25:
        raise ValueError("VRAM headroom must be 1..25 percent")
    return Limits(gpu_free_bytes=int(total_bytes * percent / 100))


def monitor_memory(stop: threading.Event, session: Session | None = None) -> None:
    """Terminate the owned process if memory reserves or sensors fail."""
    import torch

    while not stop.is_set():
        try:
            sample = sample_memory()
            free, total = torch.cuda.mem_get_info(0)
            sample["gpu_free_bytes"] = min(sample["gpu_free_bytes"], free)
            reason = violation(sample, memory_limits(total))
            if reason:
                raise RuntimeError(f"Memory safety reserve exceeded: {reason}")
            emit(memory=dict(gpu_free_bytes=sample["gpu_free_bytes"], gpu_total_bytes=total,
                             headroom_percent=float(os.environ.get("FISH_DRAMA3_WEB_HEADROOM_PERCENT", "3"))))
        except Exception as error:
            emit(status="error", message="Memory safety monitor stopped the model", error=str(error))
            # A native kernel cannot be interrupted safely in another thread.
            # Process exit releases both the native CUDA and PyTorch resources.
            os._exit(2)
        if session is not None and session.bridge is not None:
            try:
                emit(performance=session.bridge.profile_summary())
            except Exception:
                pass  # Optional performance telemetry must never stop synthesis.
        stop.wait(1.0)


def integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def generation_options(request: dict[str, Any]) -> argparse.Namespace:
    """Validate independently so direct worker callers share the UI contract."""
    args = argparse.Namespace(**request)
    args.text = request.get("text", "")
    if not isinstance(args.text, str) or not args.text.strip():
        raise ValueError("Supply nonempty text")
    args.seed = integer(request.get("seed", 123), "seed", -(2**63), 2**64 - 1)
    mode = request.get("generation_mode", "single")
    maximum = 65536 if mode == "chunked" else 8192
    args.max_frames = integer(request.get("max_frames", 512), "max_frames", 1, maximum)
    args.top_k = integer(request.get("top_k", 30), "top_k", 1, 2**31 - 1)
    args.chunk_length = integer(request.get("chunk_length", 200), "chunk_length", 0, 2000)
    if 0 < args.chunk_length < 32:
        raise ValueError("chunk_length must be 0 or 32..2000")
    args.generation_mode = request.get("generation_mode", "single")
    if args.generation_mode not in ("single", "chunked"):
        raise ValueError("generation_mode must be single or chunked")
    if args.generation_mode == "single":
        args.chunk_length = 0
    elif args.chunk_length == 0:
        raise ValueError("Smart segments require a passage length of 32..2000 characters")
    args.max_context_rows = 10240
    args.temperature, args.top_p = request.get("temperature", .7), request.get("top_p", .7)
    for name in ("temperature", "top_p"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("Temperature must be positive and top_p must be (0,1]")
    args.allow_partial = request.get("allow_partial", False)
    if not isinstance(args.allow_partial, bool):
        raise ValueError("allow_partial must be boolean")
    args.reference_text = request.get("reference_text") or None
    args.reference_audio = Path(request["reference_audio"]) if request.get("reference_audio") else None
    if bool(args.reference_audio) != bool(args.reference_text):
        raise ValueError("Reference audio and transcript must be supplied together")
    if args.reference_text is not None and not isinstance(args.reference_text, str):
        raise ValueError("Reference transcript must be text")
    args.output = Path(request["output"])
    if not args.output.is_absolute() or args.output.suffix.lower() != ".wav":
        raise ValueError("Output must be an absolute WAV path")
    if args.reference_audio and not args.reference_audio.is_absolute():
        raise ValueError("Reference audio must be an absolute path")
    if request.get("output_format", "wav") not in ("wav", "mp3"):
        raise ValueError("output_format must be wav or mp3")
    return args


class Session:
    """Own one model and codec until the subprocess is unloaded."""

    def __init__(self, request: dict[str, Any]) -> None:
        self.model = self.bridge = self.codec = None
        self.bundle, self.codec_path = Path(request["bundle"]), Path(request["codec"])
        self.threads = integer(request.get("threads", 8), "threads", 1, 256)
        self.gpu_layers = integer(request.get("gpu_expert_layers", 16), "gpu_expert_layers", 0, 40)
        from fish_drama3.web_config import placement_options
        self.placement_options = placement_options(request)
        self.placement: dict[str, Any] = {}
        self.identity: dict[str, Any] = {}

    def load(self) -> None:
        import torch
        from fish_drama3.hybrid import load_model_placed

        preflight()
        emit(phase="verifying", message="Verifying model and codec SHA-256 hashes…")
        self.identity = verify_assets(self.bundle, self.codec_path)
        os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
        os.environ.setdefault("FISH_DRAMA3_PROFILE_EXPERTS", "1")
        torch.set_num_threads(self.threads)
        started = time.perf_counter()
        emit(phase="loading", message="Loading dense weights, reserving caches and placing expert layers…")
        self.codec = CodecBridge(self.codec_path)
        selected = None if self.placement_options["placement_mode"] == "auto" else self.gpu_layers
        self.model, self.bridge, self.placement = load_model_placed(
            self.bundle, threads=self.threads, gpu_layers=selected,
            headroom_percent=self.placement_options["headroom_percent"], cache_capacity=10240)
        self.gpu_layers = self.placement["gpu_expert_layers"]
        self.load_seconds = time.perf_counter() - started
        preflight()

    def close(self) -> None:
        try:
            if self.bridge is not None:
                self.bridge.close()
        finally:
            if self.codec is not None:
                self.codec.close()
            self.model = self.bridge = self.codec = None

    def describe(self) -> dict[str, Any]:
        return dict(bundle=str(self.bundle), codec=str(self.codec_path), threads=self.threads,
                    gpu_expert_layers=self.gpu_layers, identity=self.identity,
                    load_seconds=self.load_seconds, placement=self.placement,
                    **self.placement_options)

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        import torch
        from fish_drama3.generation import generate_text, validate_drama3_text

        args = generation_options(request)
        from fish_drama3.web_planning import generation_plan
        generation_plan(vars(args))
        validate_drama3_text(args.text, "text")
        if args.reference_text:
            validate_drama3_text(args.reference_text, "reference")
        if args.output.exists() or args.output.with_suffix(".json").exists():
            raise FileExistsError("Refusing to replace an existing result")
        preflight()
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        if hasattr(self.bridge, "profile_summary"):
            self.bridge.profile_summary(reset=True)
        started = time.perf_counter()
        texts, codes, source = [], [], None
        if args.reference_audio:
            emit(phase="encoding_reference", message="Encoding your reference voice…")
            samples, source = reference_audio(args.reference_audio)
            codes = [torch.from_numpy(self.codec.encode(samples, self.threads).astype(np.int64))]
            texts = [args.reference_text]
        with torch.inference_mode():
            emit(phase="prefill", message="Reading the script and voice context…")
            frames, reason, segments = generate_text(
                self.model, args, texts, codes,
                progress=lambda data: emit(phase="generating", progress=data))
            if reason != "eos" and not args.allow_partial:
                raise RuntimeError("Frame limit reached; no complete WAV published. "
                                   "Use smart segments with a larger total frame budget, "
                                   "or shorten the script. One-shot is limited to 8192 frames "
                                   "and may still fail to finish. Your script and settings are preserved.")
            if not len(frames):
                raise RuntimeError("Model returned no acoustic frames")
            generation_seconds = time.perf_counter() - started
            emit(phase="decoding", message=f"Decoding {len(frames)} acoustic frames into audio…")
            decode_started = time.perf_counter()
            audio = self.decode_audio(frames.numpy())
            decode_seconds = time.perf_counter() - decode_started
        if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
            raise RuntimeError("Invalid or out-of-range waveform; no clipped WAV published")
        metrics = self.metrics(request, args, reason, segments, source)
        metrics.update(frames=len(frames), audio_seconds=len(audio) / 44100,
                       generation_seconds=generation_seconds,
                       decode_seconds=decode_seconds,
                       synthesis_seconds=time.perf_counter() - started,
                       peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),
                       peak_abs=float(np.max(np.abs(audio))))
        if hasattr(self.bridge, "profile_summary"):
            profile = self.bridge.profile_summary()
            if isinstance(profile, dict):
                metrics["expert_profile"] = profile
        with output_claim(args.output):
            emit(phase="exporting", message="Saving the original WAV and generation details…")
            publish_wave(args.output, audio, metrics)
        return dict(output=str(args.output), metrics=metrics)

    def decode_audio(self, rows: np.ndarray) -> np.ndarray:
        """Decode one continuous stream with bounded codec context memory."""
        if len(rows) <= 2048:
            emit(phase="decoding", progress=dict(stage_completed=0, stage_total=1))
            audio = self.codec.decode(rows, self.threads)
            emit(phase="decoding", progress=dict(stage_completed=1, stage_total=1))
            return audio
        pieces = []
        for start in range(0, len(rows), 1792):
            emit(phase="decoding", message=f"Decoding audio window {start // 1792 + 1} "
                 f"of {(len(rows) + 1791) // 1792}…",
                 progress=dict(stage_completed=start // 1792,
                               stage_total=(len(rows) + 1791) // 1792))
            context = max(0, start - 256)
            stop = min(start + 1792, len(rows))
            audio = self.codec.decode(rows[context:stop], self.threads)
            pieces.append(audio[(start - context) * 2048:])
            emit(phase="decoding", progress=dict(stage_completed=start // 1792 + 1,
                 stage_total=(len(rows) + 1791) // 1792))
        return np.concatenate(pieces)

    def metrics(self, request: dict[str, Any], args: argparse.Namespace,
                reason: str, segments: list[dict], source: dict | None) -> dict[str, Any]:
        settings = {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != "action"}
        settings.update(bundle=str(self.bundle), codec=str(self.codec_path),
                        threads=self.threads, gpu_expert_layers=self.gpu_layers,
                        **self.placement_options,
                        output_format=request.get("output_format", "wav"),
                        mp3_bitrate=request.get("mp3_bitrate", 192))
        return dict(identity=self.identity, settings=settings, text=args.text, seed=args.seed,
                    sampling=dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k),
                    gpu_expert_layers=self.gpu_layers, threads=self.threads,
                    max_frames=args.max_frames, chunk_length=args.chunk_length,
                    allow_partial=args.allow_partial, reference=source,
                    reference_text=args.reference_text, segments=segments,
                    finish_reason=reason, complete=reason == "eos", load_seconds=self.load_seconds,
                    runtime=f"hybrid-windows-web-{__version__}")


def fatal_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(term in message for term in ("cuda", "out of memory", "cublas", "device-side assert"))


def handle_requests(session: Session) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Worker request must be a JSON object")
            if request.get("action") == "unload":
                return
            if request.get("action") != "generate":
                raise ValueError("Expected generate or unload action")
            result = session.generate(request)
            emit(status="ready", message="Generation complete", result=result)
        except Exception as error:
            fatal = fatal_error(error)
            emit(status="error" if fatal else "ready", message="Generation failed", error=str(error))
            if fatal:
                raise


def main() -> None:
    session = None
    stop = threading.Event()
    monitor = None
    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict) or request.get("action") != "load":
            raise ValueError("First worker request must load a model")
        from fish_drama3.web_config import placement_options
        os.environ["FISH_DRAMA3_WEB_HEADROOM_PERCENT"] = str(placement_options(request)["headroom_percent"])
        with guarded_job_lock():
            try:
                session = Session(request)
                monitor = threading.Thread(target=monitor_memory, args=(stop, session),
                                           name="studio-memory-monitor", daemon=True)
                monitor.start()
                session.load()
                emit(status="ready", phase="ready", message="Model loaded", model=session.describe())
                handle_requests(session)
            finally:
                stop.set()
                if monitor is not None:
                    monitor.join(timeout=6)
                if session is not None:
                    session.close()
    except Exception as error:
        emit(status="error", message="Worker stopped", error=str(error))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
