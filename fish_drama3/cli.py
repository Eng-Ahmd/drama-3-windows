"""Offline Windows synthesis using the quantized model and native codec."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import soundfile as sf

from fish_drama3.codec_adapter import CodecBridge, sha_file
from fish_drama3 import __version__


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"fish-drama3 {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synth")
    synth.add_argument("--bundle", type=Path, required=True)
    synth.add_argument("--codec", type=Path, required=True)
    synth.add_argument("--text", required=True)
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--reference-audio", type=Path)
    synth.add_argument("--reference-text")
    synth.add_argument("--gpu-expert-layers", type=int, default=12)
    synth.add_argument("--threads", type=int, default=8)
    synth.add_argument("--max-frames", type=int, default=512)
    synth.add_argument("--chunk-length", type=int, default=200,
                       help="Maximum text characters per segment; 0 disables segmentation")
    synth.add_argument("--seed", type=int, default=123)
    synth.add_argument("--temperature", type=float, default=0.7)
    synth.add_argument("--top-p", type=float, default=0.7)
    synth.add_argument("--top-k", type=int, default=30)
    synth.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".wav":
        parser.error("Output filename must end in .wav")
    if not 0 <= args.gpu_expert_layers <= 16 or not 1 <= args.threads <= 256:
        parser.error("GPU expert layers must be 0..16; threads must be 1..256")
    if not 1 <= args.max_frames <= 2048 or not args.text.strip():
        parser.error("Supply nonempty text and --max-frames 1..2048")
    if args.chunk_length != 0 and not 32 <= args.chunk_length <= 2000:
        parser.error("Chunk length must be 0 or 32..2000 characters")
    if bool(args.reference_audio) != bool(args.reference_text):
        parser.error("Reference audio and transcript must be supplied together")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive")
    if not 0 < args.top_p <= 1 or args.top_k < 1:
        parser.error("Top-p must be (0,1] and top-k positive")
    return args


def verify_assets(bundle: Path, codec: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    codec_manifest = json.loads((codec.parent / "manifest.json").read_text(encoding="utf-8"))
    model_hash, codec_hash = sha_file(bundle / "model.gguf"), sha_file(codec)
    if model_hash != manifest.get("model_sha256") or codec_hash != codec_manifest.get("sha256"):
        raise ValueError("Model or codec SHA-256 does not match its manifest")
    return dict(model_sha256=model_hash, codec_sha256=codec_hash)


def reference_audio(path: Path) -> tuple[np.ndarray, dict]:
    from scipy.signal import resample_poly

    with sf.SoundFile(path) as audio_file:
        if not 1 <= audio_file.channels <= 8 or not 8000 <= audio_file.samplerate <= 192000:
            raise ValueError("Reference audio supports 1..8 channels at 8..192 kHz")
        if audio_file.frames < 1 or audio_file.frames / audio_file.samplerate > 60:
            raise ValueError("Reference audio must have duration greater than zero and at most 60 seconds")
        rate = audio_file.samplerate
        values = audio_file.read(dtype="float32", always_2d=True).mean(axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite reference audio")
    if rate != 44100:
        divisor = math.gcd(rate, 44100)
        values = resample_poly(values, 44100 // divisor, rate // divisor).astype(np.float32)
    return values, dict(sha256=sha_file(path), source_rate=rate, target_rate=44100,
                        resampler="scipy.signal.resample_poly/default" if rate != 44100 else "none")


@contextmanager
def output_claim(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("Output is already claimed; choose another path or inspect a stale .lock") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid(), output=str(path.resolve())), stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def publish_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        source.rename(destination)  # Windows rename fails if destination exists.
    else:
        os.link(source, destination)


def publish_wave(path: Path, audio: np.ndarray, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    temporary = path.with_name(path.name + f".{suffix}.partial")
    report = path.with_suffix(".json")
    report_temporary = report.with_name(report.name + f".{suffix}.partial")
    wave_created = False
    try:
        sf.write(temporary, audio, 44100, format="WAV", subtype="PCM_16")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        with report_temporary.open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        wave_identity = temporary.stat()
        publish_no_replace(temporary, path)
        wave_created = True
        publish_no_replace(report_temporary, report)
    except BaseException:
        if wave_created and path.exists():
            current = path.stat()
            if (current.st_dev, current.st_ino) == (wave_identity.st_dev, wave_identity.st_ino):
                path.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)


def synthesize(args: argparse.Namespace) -> dict:
    import torch
    from fish_drama3.hybrid import load_model
    from fish_drama3.generation import generate_text, validate_drama3_text

    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to replace an existing audio result")
    validate_drama3_text(args.text, "text")
    if args.reference_text:
        validate_drama3_text(args.reference_text, "reference")
    identity = verify_assets(args.bundle, args.codec)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    with CodecBridge(args.codec) as codec:
        reference, source = None, None
        if args.reference_audio:
            samples, source = reference_audio(args.reference_audio)
            reference = torch.from_numpy(codec.encode(samples, args.threads).astype(np.int64))
        load_started = time.perf_counter()
        model, bridge = load_model(args.bundle, threads=args.threads, gpu_layers=args.gpu_expert_layers)
        load_seconds = time.perf_counter() - load_started
        try:
            texts, codes = ([args.reference_text], [reference]) if reference is not None else ([], [])
            started = time.perf_counter()
            frames, reason, segments = generate_text(model, args, texts, codes)
            if reason != "eos" and not args.allow_partial:
                raise RuntimeError("Frame limit reached; no complete WAV published. Increase --max-frames")
            if len(frames) == 0:
                raise RuntimeError("Model returned no acoustic frames")
            generated = time.perf_counter() - started
            audio = codec.decode(frames.numpy(), args.threads)
            metrics = dict(identity=identity, text=args.text, seed=args.seed,
                sampling=dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k),
                gpu_expert_layers=args.gpu_expert_layers, threads=args.threads,
                finish_reason=reason, complete=reason == "eos", frames=len(frames),
                chunk_length=args.chunk_length, segments=segments,
                audio_seconds=len(audio) / 44100, generation_seconds=generated,
                synthesis_seconds=time.perf_counter() - started, load_seconds=load_seconds,
                reference=source, peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),
                peak_abs=float(np.max(np.abs(audio))), runtime=f"hybrid-windows-{__version__}")
            if metrics["peak_abs"] > 1:
                raise RuntimeError("Waveform exceeds PCM full scale; no clipped WAV published")
            publish_wave(args.output, audio, metrics)
            return metrics
        finally:
            bridge.close()


def main() -> None:
    stop = thread = None
    try:
        args = parse_options()
        from fish_drama3.watchdog import guarded_job_lock
        lease = nullcontext() if os.environ.get("FISH_DRAMA3_WATCHDOG_ACTIVE") == "1" else guarded_job_lock()
        if os.environ.get("FISH_DRAMA3_WATCHDOG_ACTIVE") == "1":
            from fish_drama3.watchdog import start_child_telemetry
            stop, thread = start_child_telemetry()
        with lease, output_claim(args.output):
            print(json.dumps(synthesize(args), indent=2), flush=True)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"fish-drama3: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        print("fish-drama3: cancelled", file=sys.stderr)
        raise SystemExit(130)
    finally:
        if stop is not None:
            stop.set()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
