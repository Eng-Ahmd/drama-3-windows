"""Compare isolated CPU codec builds on saved real acoustic-code fixtures.

No model synthesis or GPU allocation is performed. The runtime job lease blocks
the benchmark while an active guarded generation owns the machine. Build the
candidate in a separate directory; enable only the candidate CMake options under
evaluation. Do not replace a DLL loaded by an active studio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fish_drama3.codec_adapter import CodecBridge, prepare_codes
from fish_drama3.watchdog import guarded_job_lock


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[32])
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--check-encode", action="store_true",
                        help="Also require exact reference-encoder code parity")
    parser.add_argument("--stream-frames", type=int, default=0,
                        help="Also compare unchanged 1792-frame/256-context long decoding")
    parser.add_argument("--study-frames", type=int, default=0,
                        help="Compare real long-stream window sizes using --baseline only")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[256, 512, 1024, 1792])
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reference-audio", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.repeats <= 10:
        parser.error("repeats must be 0..10 (0 measures the first decode only)")
    if any(not 1 <= count <= 4096 for count in args.frames):
        parser.error("frame counts must be 1..4096")
    if any(not 1 <= count <= 256 for count in args.threads):
        parser.error("thread counts must be 1..256")
    if args.stream_frames and not 2048 < args.stream_frames <= 8192:
        parser.error("stream frames must be 2049..8192")
    if args.study_frames and (not 2048 < args.study_frames <= 8192 or not args.baseline):
        parser.error("window study needs --baseline and 2049..8192 real frames")
    if any(not 1 <= size <= 1792 for size in args.window_sizes):
        parser.error("window sizes must be 1..1792 (plus 256 context frames)")
    if not args.worker and not args.study_frames and (not args.baseline or not args.candidate):
        parser.error("supply both --baseline and --candidate")
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("parity tolerances must be finite and nonnegative")
    return args


def timed_decode(bridge: CodecBridge, rows: np.ndarray,
                 threads: int, windowed: bool = False) -> tuple[np.ndarray, float]:
    if windowed:
        return window_decode(bridge, rows, 1792, threads)
    started = time.perf_counter()
    audio = bridge.decode(rows, threads)
    return audio, time.perf_counter() - started


def case(bridge: CodecBridge, rows: np.ndarray, threads: int,
         args: argparse.Namespace, windowed: bool = False) -> dict[str, Any]:
    prefix = "stream" if windowed else "frames"
    identifier = f"{prefix}-{len(rows)}-threads-{threads}"
    first, first_seconds = timed_decode(bridge, rows, threads, windowed)
    times: list[float] = []
    repeat_max_abs = 0.0
    for _ in range(args.repeats):
        repeated, elapsed = timed_decode(bridge, rows, threads, windowed)
        if repeated.shape != first.shape:
            raise RuntimeError("Repeated codec output sample count changed")
        repeat_max_abs = max(repeat_max_abs, float(np.max(np.abs(first - repeated))))
        if not np.allclose(first, repeated, atol=args.atol, rtol=args.rtol):
            raise RuntimeError("Repeated codec output exceeded waveform parity tolerance")
        times.append(elapsed)
    np.save(args.output.parent / f"{args.output.stem}-{identifier}.npy", first)
    duration = len(first) / 44100
    measured = statistics.median(times) if times else first_seconds
    return dict(id=identifier, frames=len(rows), threads=threads, samples=len(first),
                first_seconds=first_seconds, repeat_seconds=times,
                median_repeat_seconds=measured,
                timing_basis="repeat" if times else "first",
                real_time_factor=measured / duration,
                repeat_max_abs=repeat_max_abs,
                waveform_sha256=hashlib.sha256(first.tobytes()).hexdigest())


def seam_case(bridge: CodecBridge, rows: np.ndarray,
              args: argparse.Namespace) -> dict[str, Any] | None:
    """Compare unchanged context-window assembly across the two builds."""
    if len(rows) < 48:
        return None
    left = bridge.decode(rows[:32], args.threads[0])
    right = bridge.decode(rows[16:48], args.threads[0])
    # Compare the same midpoint overlap trim in both builds. This does not
    # establish parity with a whole-context decoder or certify seam audibility.
    midpoint = 24 * 2048
    joined = np.concatenate([left[:midpoint], right[8 * 2048:]])
    path = args.output.parent / f"{args.output.stem}-seam.npy"
    np.save(path, joined)
    return dict(samples=len(joined), seam_index=midpoint,
                boundary_step=float(joined[midpoint] - joined[midpoint - 1]))


def run_worker(args: argparse.Namespace) -> None:
    rows = np.load(args.codes, allow_pickle=False)
    prepare_codes(rows)
    if max(*args.frames, args.stream_frames) > len(rows):
        raise ValueError("Fixture is shorter than the requested frame count")
    with CodecBridge(args.codec, args.worker) as bridge:
        cases = [case(bridge, rows[:frames], threads, args)
                 for frames in dict.fromkeys(args.frames)
                 for threads in dict.fromkeys(args.threads)]
        if args.stream_frames:
            cases += [case(bridge, rows[:args.stream_frames], threads, args, True)
                      for threads in dict.fromkeys(args.threads)]
        seams = seam_case(bridge, rows, args)
        encoded = None
        if args.check_encode:
            audio = (np.load(args.reference_audio, allow_pickle=False)
                     if args.reference_audio else
                     bridge.decode(rows[:min(32, len(rows))], args.threads[0]))
            np.save(args.output.parent / f"{args.output.stem}-reference.npy", audio)
            codes = bridge.encode(audio, args.threads[0])
            np.save(args.output.parent / f"{args.output.stem}-encoded.npy", codes)
            encoded = dict(shape=list(codes.shape),
                           sha256=hashlib.sha256(codes.tobytes()).hexdigest())
    report = dict(library=str(args.worker.resolve()),
                  library_sha256=hashlib.sha256(args.worker.read_bytes()).hexdigest(),
                  codec=str(args.codec.resolve()), fixture=str(args.codes.resolve()),
                  fixture_sha256=hashlib.sha256(args.codes.read_bytes()).hexdigest(),
                  cases=cases, seam=seams, encoded=encoded)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def launch_worker(args: argparse.Namespace, name: str, library: Path) -> Path:
    output = args.output.parent / f"{args.output.stem}-{name}.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(library),
               "--codec", str(args.codec), "--codes", str(args.codes),
               "--output", str(output), "--frames", *map(str, args.frames),
               "--threads", *map(str, args.threads), "--repeats", str(args.repeats),
               "--atol", str(args.atol), "--rtol", str(args.rtol)]
    if args.stream_frames:
        command.extend(["--stream-frames", str(args.stream_frames)])
    if args.check_encode:
        command.append("--check-encode")
        if name == "candidate":
            reference = args.output.parent / f"{args.output.stem}-baseline-reference.npy"
            command.extend(["--reference-audio", str(reference)])
    subprocess.run(command, check=True)
    return output


def waveform_comparison(baseline: Path, candidate: Path,
                        args: argparse.Namespace) -> dict[str, Any]:
    before = np.load(baseline, allow_pickle=False)
    after = np.load(candidate, allow_pickle=False)
    if before.shape != after.shape:
        return dict(passed=False, baseline_shape=before.shape, candidate_shape=after.shape)
    difference = before.astype(np.float64) - after.astype(np.float64)
    return dict(passed=bool(np.allclose(before, after, atol=args.atol, rtol=args.rtol)),
                bit_identical=bool(np.array_equal(before, after)),
                max_abs=float(np.max(np.abs(difference))),
                rms=float(np.sqrt(np.mean(difference**2))))


def compare_reports(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    before, after = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    comparisons = []
    for baseline, candidate in zip(before["cases"], after["cases"], strict=True):
        identifier = baseline["id"]
        wave_paths = [path.parent / f"{path.stem}-{identifier}.npy" for path in paths]
        parity = waveform_comparison(*wave_paths, args)
        comparisons.append(dict(id=identifier, parity=parity,
                                baseline=baseline, candidate=candidate,
                                repeat_speedup=baseline["median_repeat_seconds"] /
                                candidate["median_repeat_seconds"]))
    seam = None
    if before["seam"] and after["seam"]:
        seam_paths = [path.parent / f"{path.stem}-seam.npy" for path in paths]
        seam = waveform_comparison(*seam_paths, args)
    passed = all(item["parity"]["passed"] for item in comparisons)
    encoder_parity = None
    if args.check_encode:
        encoded = [np.load(path.parent / f"{path.stem}-encoded.npy",
                           allow_pickle=False) for path in paths]
        encoder_parity = bool(np.array_equal(*encoded))
    return dict(passed=passed and (seam is None or seam["passed"])
                       and encoder_parity is not False,
                encoder_codes_identical=encoder_parity,
                cases=comparisons, window_assembly_parity=seam,
                limitation="Compares runtime builds, not source-model waveform fidelity. "
                           "Window assembly parity does not certify audible seamlessness.")


def window_decode(bridge: CodecBridge, rows: np.ndarray, size: int,
                  threads: int) -> tuple[np.ndarray, float]:
    pieces = []
    started = time.perf_counter()
    for start in range(0, len(rows), size):
        context = max(0, start - 256)
        stop = min(start + size, len(rows))
        audio = bridge.decode(rows[context:stop], threads)
        pieces.append(audio[(start - context) * 2048:])
    return np.concatenate(pieces), time.perf_counter() - started


def window_study(args: argparse.Namespace) -> None:
    """Quantify changed-window output, including errors around each new seam."""
    rows = np.load(args.codes, allow_pickle=False)[:args.study_frames]
    if len(rows) != args.study_frames:
        raise ValueError("Window study requires sufficient real saved code frames")
    prepare_codes(rows)
    comparisons = []
    report = dict(partial=True, frames=len(rows), threads=args.threads[0], context=256,
                  baseline_window=1792, comparisons=comparisons)
    with CodecBridge(args.codec, args.baseline) as bridge:
        baseline, seconds = window_decode(bridge, rows, 1792, args.threads[0])
        np.save(args.output.with_suffix(".baseline.npy"), baseline)
        report["baseline_seconds"] = seconds
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for size in dict.fromkeys(args.window_sizes):
            if size == 1792:
                continue
            wave, elapsed = window_decode(bridge, rows, size, args.threads[0])
            np.save(args.output.with_suffix(f".window-{size}.npy"), wave)
            difference = wave.astype(np.float64) - baseline
            rms = float(np.sqrt(np.mean(difference**2)))
            edge = np.zeros(len(wave), dtype=bool)
            for frame in range(size, len(rows), size):
                index = frame * 2048
                edge[max(0, index - 2048):index + 2048] = True
            comparisons.append(dict(window=size, seconds=elapsed, speedup=seconds / elapsed,
                bit_identical=bool(np.array_equal(wave, baseline)), rms=rms,
                max_abs=float(np.max(np.abs(difference))),
                relative_rms=rms / max(float(np.sqrt(np.mean(baseline**2))), 1e-12),
                edge_rms=float(np.sqrt(np.mean(difference[edge]**2))),
                interior_rms=float(np.sqrt(np.mean(difference[~edge]**2)))))
            print(json.dumps(comparisons[-1]), flush=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["partial"] = False
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_options()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_worker(args)
        return
    if args.study_frames:
        if args.output.exists():
            raise FileExistsError("Choose a new window-study report path")
        with guarded_job_lock():
            window_study(args)
        return
    if args.baseline.resolve() == args.candidate.resolve():
        raise ValueError("Use distinct baseline and candidate build directories")
    if args.output.exists():
        raise FileExistsError("Choose a new report path to preserve earlier evidence")
    with guarded_job_lock():
        paths = [launch_worker(args, name, library) for name, library in
                 [("baseline", args.baseline), ("candidate", args.candidate)]]
    report = compare_reports(paths, args)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(report=str(args.output), passed=report["passed"])))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
