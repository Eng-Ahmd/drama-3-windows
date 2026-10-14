"""Frozen preview and guarded soak checks; production qualification is separate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from fish_drama3.codec_adapter import CodecBridge, DESTINATION, prepare_codes, sha_file
from fish_drama3.cli import verify_assets

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/qualification/cases.jsonl"
REF_TEXT = "We asked over twenty different people, and they all said it was his."


@dataclass(frozen=True)
class Criteria:
    eos_required: bool = True
    max_nonfinite: int = 0
    max_invalid_codes: int = 0
    drift_bytes: int = 256 * 2**20
    warmup_requests: int = 12
    production_seconds: int = 7200
    production_requests: int = 200
    production_failure_cycles: int = 50
    production_corpus_cases: int = 100


def watchdog_thresholds() -> dict:
    values = dict(gpu_free_gib=float(os.environ.get("FISH_DRAMA3_WATCHDOG_GPU_GIB", "1")),
                  host_free_gib=float(os.environ.get("FISH_DRAMA3_WATCHDOG_RAM_GIB", "8")),
                  max_commit_percent=float(os.environ.get("FISH_DRAMA3_WATCHDOG_COMMIT_PERCENT", "90")))
    if not all(np.isfinite(value) and value > 0 for value in values.values()):
        raise ValueError("Invalid watchdog thresholds")
    return values


def append_event(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Corrupt completed checkpoint record")
            # A killed append can leave only the final record truncated.
            with path.open("r+b") as handle:
                handle.truncate(sum(len(item) for item in lines[:index]))
    return events


def frozen_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({case["id"] for case in cases}) != len(cases) or not cases:
        raise ValueError("Cases need nonempty unique IDs")
    for case in cases:
        if not isinstance(case["text"], str) or not 1 <= case["max_frames"] <= 2048:
            raise ValueError("Invalid frozen case")
        if case.get("text_repeat", 1) < 1:
            raise ValueError("Invalid text repetition")
    return cases


def scheduled_case(cases: list[dict], index: int, mode: str) -> dict:
    if mode == "preview":
        # Replay short anchor immediately after long and all boundary failures.
        schedule = cases + [cases[0]]
        return schedule[index]
    positive = [case for case in cases if not case.get("expect_reject")]
    negative = [case for case in cases if case.get("expect_reject")]
    if index < len(cases):
        return cases[index]
    step = index - len(cases)
    if step % 6 == 4:
        return negative[(step // 6) % len(negative)]
    if step % 6 == 5:
        return positive[0]
    return positive[step % len(positive)]


def memory_snapshot(capacity: int) -> dict:
    memory = psutil.Process().memory_info()
    private = getattr(memory, "private", None)
    result = dict(private_bytes=private, rss_bytes=memory.rss,
                  host_free_bytes=psutil.virtual_memory().available,
                  cache_capacity=int(capacity))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(cuda_free_bytes=free, cuda_total_bytes=total,
                      cuda_used_bytes=total - free,
                      torch_allocated_bytes=torch.cuda.memory_allocated(),
                      torch_reserved_bytes=torch.cuda.memory_reserved())
    return result


def memory_drift(baseline: dict, current: dict, limit: int) -> dict:
    if baseline["cache_capacity"] != current["cache_capacity"]:
        return dict(status="declared_cache_growth", passed=None)
    keys = ["private_bytes", "torch_allocated_bytes", "cuda_used_bytes"]
    unavailable = [key for key in keys if baseline.get(key) is None or current.get(key) is None]
    deltas = {key: current[key] - baseline[key] for key in keys if key not in unavailable}
    return dict(status="measured", deltas=deltas, unavailable=unavailable,
                passed=not unavailable and all(value <= limit for value in deltas.values()))


def source_identity(options: argparse.Namespace) -> dict:
    manifest = json.loads((options.bundle / "manifest.json").read_text(encoding="utf-8"))
    codec_manifest = json.loads((options.codec.parent / "manifest.json").read_text(encoding="utf-8"))
    build = "drama3-cuda" if options.gpu_expert_layers else "drama3-inspect"
    binaries = list((ROOT / "builds" / build / "Release").glob("*.dll"))
    binaries += list((ROOT / "builds/drama3-inspect/Release").glob("*.dll"))
    return dict(bundle_manifest_sha256=sha_file(options.bundle / "manifest.json"),
                declared_model_sha256=manifest.get("model_sha256"),
                codec_manifest_sha256=sha_file(options.codec.parent / "manifest.json"),
                declared_codec_sha256=codec_manifest.get("sha256"),
                corpus_sha256=sha_file(options.corpus),
                runner_sha256=sha_file(Path(__file__)),
                hybrid_source_sha256=sha_file(ROOT / "fish_drama3/hybrid.py"),
                native_hashes={str(path.relative_to(ROOT)): sha_file(path) for path in binaries},
                reference_codes_sha256=sha_file(ROOT / "models/comparison/en0-codes.npy"),
                reference_transcript=REF_TEXT,
                sampling=dict(temperature=0.7, top_p=0.7, top_k=30),
                mode=options.mode, gpu_expert_layers=options.gpu_expert_layers,
                threads=options.threads, criteria=asdict(Criteria()),
                watchdog_thresholds=watchdog_thresholds(),
                requested_soak_seconds=options.soak_seconds,
                requested_min_requests=options.min_requests,
                requested_failure_cycles=options.failure_cycles)


def initialize_run(options: argparse.Namespace) -> tuple[dict, list[dict]]:
    options.output.mkdir(parents=True, exist_ok=True)
    manifest_path = options.output / "run.json"
    events = read_events(options.output / "requests.jsonl")
    if options.resume and any(event.get("event") == "completed" and not event.get("passed")
                              for event in events):
        raise ValueError("A failed run cannot be resumed; preserve it and start a new output directory")
    actual_assets = verify_assets(options.bundle, options.codec)
    identity = source_identity(options)
    identity["verified_assets"] = actual_assets
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not options.resume or manifest["identity"] != identity:
            raise ValueError("Resume requires identical frozen source/options/criteria")
    else:
        manifest = dict(identity=identity, status="running", qualified=False,
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        machine=dict(platform=platform.platform(), cpu=platform.processor(),
                                     host_ram=psutil.virtual_memory().total),
                        versions={name: importlib.metadata.version(name) for name in
                                  ["torch", "numpy", "psutil"]}, segments=[])
    manifest["segments"].append(dict(started_utc=datetime.now(timezone.utc).isoformat(),
                                      watchdog=os.environ.get("FISH_DRAMA3_WATCHDOG_ACTIVE") == "1"))
    manifest["status"] = "running"
    (options.output / "runner-source.py").write_bytes(Path(__file__).read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, events


def case_prompt(case: dict, model):
    from fish_drama3.hybrid import build_drama3_prompt

    text = case["text"] * case.get("text_repeat", 1)
    if case.get("reference"):
        path = ROOT / "models/comparison/en0-codes.npy"
        codes = np.load(path, allow_pickle=False)
        return build_drama3_prompt(text, model.tokenizer, model.config,
                                  [REF_TEXT], [torch.from_numpy(codes.astype(np.int64))])
    return build_drama3_prompt(text, model.tokenizer, model.config)


def execute_case(case: dict, model, codec: CodecBridge) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    from fish_drama3.hybrid import generate_drama3

    torch.manual_seed(case["seed"])
    started = time.perf_counter()
    try:
        prompt = case_prompt(case, model)
        if len(prompt) + case["max_frames"] > model.config.text.max_position_embeddings:
            raise ValueError("Generation exceeds prompt row budget")
    except ValueError as error:
        if not case.get("expect_reject"):
            raise
        return dict(passed=True, rejected=True, error=str(error),
                    prompt_seconds=time.perf_counter() - started), None, None
    if case.get("expect_reject"):
        return dict(passed=False, rejected=False, error="Expected invalid request accepted"), None, None
    prompt_seconds = time.perf_counter() - started
    started = time.perf_counter()
    frames, reason = generate_drama3(model, prompt, max_new_tokens=case["max_frames"],
                                     temperature=0.7, top_p=0.7, top_k=30)
    rows = frames.numpy()
    generation_seconds = time.perf_counter() - started
    prepare_codes(rows)
    started = time.perf_counter()
    audio = codec.decode(rows)
    codec_seconds = time.perf_counter() - started
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite waveform")
    return dict(passed=reason == "eos", rejected=False, finish_reason=reason,
                frames=len(rows), samples=len(audio), prompt_rows=len(prompt),
                prompt_seconds=prompt_seconds, generation_seconds=generation_seconds,
                codec_seconds=codec_seconds, codes_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
                e2e_rtf=(prompt_seconds + generation_seconds + codec_seconds) / (len(audio) / 44100)), rows, audio


def record_request(index: int, case: dict, options, model, codec, hashes: dict) -> dict:
    started = time.perf_counter()
    record = dict(index=index, case_id=case["id"], seed=case["seed"],
                  event="completed", segment=len(options.manifest["segments"]) - 1)
    append_event(options.output / "requests.jsonl", {**record, "event": "started"})
    rows = audio = None
    try:
        outcome, rows, audio = execute_case(case, model, codec)
        record.update(outcome)
        key = case["id"] + ":" + str(case["seed"])
        if rows is not None:
            previous = hashes.get(key)
            record["seed_repeat_identical"] = previous is None or previous == record["codes_sha256"]
            hashes.setdefault(key, record["codes_sha256"])
            record["passed"] &= record["seed_repeat_identical"]
            retain = not (options.output / (case["id"] + ".npy")).exists() or not record["passed"]
            if retain:
                import soundfile
                stem = case["id"] if record["passed"] else f"failure-{index}"
                np.save(options.output / (stem + ".npy"), rows, allow_pickle=False)
                soundfile.write(options.output / (stem + ".wav"), audio, 44100, subtype="FLOAT")
    except Exception as error:
        record.update(passed=False, error_type=type(error).__name__, error=str(error))
    record.update(wall_seconds=time.perf_counter() - started,
                  memory=memory_snapshot(model.cache_capacity))
    return record


def run_requests(options, cases, model, codec, events) -> dict:
    completed = [event for event in events if event["event"] == "completed"]
    index = max((record["index"] for record in completed), default=-1) + 1
    hashes = {}
    for record in completed:
        if record.get("codes_sha256"):
            hashes.setdefault(record["case_id"] + ":" + str(record["seed"]), record["codes_sha256"])
    positive = recovery = local_count = 0
    pending_failure = False
    baselines = {}
    warm_started = None
    failed = any(not record.get("passed") for record in completed)
    elapsed = 0
    while not failed and (options.mode == "soak" or index < len(cases) + 1):
        case = scheduled_case(cases, index, options.mode)
        record = record_request(index, case, options, model, codec, hashes)
        local_count += 1
        if record.get("rejected"):
            pending_failure = True
        elif record.get("passed"):
            positive += 1
            recovery += int(pending_failure)
            pending_failure = False
        if local_count >= Criteria().warmup_requests:
            warm_started = warm_started or time.monotonic()
            capacity = record["memory"]["cache_capacity"]
            baseline = baselines.setdefault(capacity, record["memory"])
            record["drift"] = memory_drift(baseline, record["memory"], Criteria().drift_bytes)
            record["passed"] &= record["drift"]["passed"] is True
        append_event(options.output / "requests.jsonl", record)
        print(json.dumps({key: record.get(key) for key in ("index", "case_id", "passed", "wall_seconds")}), flush=True)
        failed |= not record["passed"]
        index += 1
        elapsed = time.monotonic() - warm_started if warm_started else 0
        if failed or (options.mode == "soak" and elapsed >= options.soak_seconds
                      and positive >= options.min_requests and recovery >= options.failure_cycles):
            break
    return dict(preview_checks_passed=not failed, completed_requests=positive,
                negative_recovery_cycles=recovery, warm_seconds=elapsed,
                counter_scope="current continuous process segment; prior segments excluded",
                cancellation_recovery="unavailable: upstream generation has no cancel callback",
                qualified=False, production_reference="unavailable: full FP8 needs 48 GB runner",
                production_corpus="pending: preview is 12 cases, target is at least 100",
                acoustic_holdout="pending: independent 300-case holdout and listening",
                status="failed" if failed else "completed-unqualified")


def normalize_english(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    target, output = normalize_english(reference), normalize_english(hypothesis)
    if not target:
        raise ValueError("WER needs nonempty reference transcript")
    previous = list(range(len(output) + 1))
    for row, token in enumerate(target, 1):
        current = [row]
        for column, guess in enumerate(output, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (token != guess)))
        previous = current
    return previous[-1] / len(target)


def asr_compare(model_path: Path, files: list[Path], transcript: str, output: Path) -> None:
    from faster_whisper import WhisperModel

    if not model_path.is_dir():
        raise ValueError("ASR model must be an explicit local frozen directory")
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=8)
    records = []
    for path in files:
        segments, _ = model.transcribe(str(path), language="en", beam_size=5, temperature=0,
                                       vad_filter=False, condition_on_previous_text=False)
        hypothesis = " ".join(segment.text.strip() for segment in segments)
        records.append(dict(file=str(path), sha256=sha_file(path), hypothesis=hypothesis,
                            wer=word_error_rate(transcript, hypothesis)))
    output.write_text(json.dumps(dict(model=str(model_path), model_files={
        str(path.relative_to(model_path)): sha_file(path) for path in model_path.rglob("*") if path.is_file()},
        reference=transcript, normalization="NFKC lowercase Unicode word tokens",
        evaluator_version=importlib.metadata.version("faster-whisper"), results=records,
        limitation="ASR is an acoustic metric, not numerical or speaker parity"), indent=2), encoding="utf-8")


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/drama3-q4k-q8")
    parser.add_argument("--codec", type=Path, default=DESTINATION / "codec.gguf")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["preview", "soak"], default="preview")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-expert-layers", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--soak-seconds", type=int, default=7200)
    parser.add_argument("--min-requests", type=int, default=200)
    parser.add_argument("--failure-cycles", type=int, default=50)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--asr-audio", type=Path, nargs="+")
    parser.add_argument("--asr-transcript")
    options = parser.parse_args()
    if not 0 <= options.gpu_expert_layers <= 12 or not 1 <= options.threads <= 256:
        parser.error("QA accepts only 0..12 GPU expert layers and 1..256 threads")
    if min(options.soak_seconds, options.min_requests, options.failure_cycles) < 0:
        parser.error("Soak settings must be nonnegative")
    return options


def main() -> None:
    options = parse_options()
    if options.asr_model:
        if not options.asr_audio or not options.asr_transcript:
            raise ValueError("ASR needs audio files and an exact reference transcript")
        options.output.mkdir(parents=True, exist_ok=True)
        asr_compare(options.asr_model, options.asr_audio, options.asr_transcript,
                    options.output / "asr.json")
        return
    if os.environ.get("FISH_DRAMA3_WATCHDOG_ACTIVE") != "1":
        raise RuntimeError("QA GPU execution requires the external safety watchdog")
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    torch.set_num_threads(options.threads)
    cases = frozen_cases(options.corpus)
    manifest, events = initialize_run(options)
    options.manifest = manifest
    from fish_drama3.hybrid import load_model
    from fish_drama3.watchdog import start_child_telemetry
    stop, thread = start_child_telemetry()
    bridge = None
    try:
        model, bridge = load_model(options.bundle, threads=options.threads,
                                   gpu_layers=options.gpu_expert_layers)
        with CodecBridge(options.codec) as codec:
            outcome = run_requests(options, cases, model, codec, events)
        manifest.update(outcome)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (options.output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        if bridge is not None:
            bridge.close()
        stop.set()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
