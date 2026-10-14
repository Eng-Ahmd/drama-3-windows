"""Generate the three official-card comparisons with matching reference/history."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any
from pathlib import Path

import numpy as np
import torch

from fish_drama3.cli import output_claim, publish_wave, reference_audio, verify_assets
from fish_drama3.codec_adapter import CodecBridge, prepare_codes, sha_file
from fish_drama3.hybrid import build_drama3_prompt, generate_drama3, load_model

REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
CASES = [
    ("point-tags", "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.", []),
    ("context-wednesday", "The quarterly report is due on Wednesday, so please finish it by noon.", []),
    ("contextual-edit", "The quarterly report is due on Thursday, so please finish it by noon.", ["context-wednesday"]),
    ("chained-edit", "The quarterly report is due on Thursday, so please finish it by five.", ["context-wednesday", "contextual-edit"]),
]


def generate_case(model: Any, codec: CodecBridge, args: argparse.Namespace,
                  identity: dict, reference: torch.Tensor, takes: dict,
                  case: tuple[str, str, list[str]]) -> dict:
    name, text, previous = case
    history = [(takes[item]["text"], takes[item]["codes"]) for item in previous]
    prompt = build_drama3_prompt(text, model.tokenizer, model.config,
                                [REFERENCE_TEXT], [reference], history=history)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    frames, reason = generate_drama3(model, prompt, max_new_tokens=args.max_frames,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    generation_seconds = time.perf_counter() - started
    if reason != "eos" or not len(frames):
        raise RuntimeError(f"{name}: incomplete generation ({reason}); no WAV published")
    audio = codec.decode(frames.numpy(), args.threads)
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(audio).all() or peak > 1:
        raise RuntimeError(f"{name}: invalid/full-scale waveform")
    metrics = dict(identity=identity, text=text, seed=args.seed,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        finish_reason=reason, complete=True, frames=len(frames),
        generation_seconds=generation_seconds, synthesis_seconds=time.perf_counter()-started,
        audio_seconds=len(audio)/44100, previous_takes=previous, peak_abs=peak)
    path = args.output / f"{name}.wav"
    with output_claim(path):
        publish_wave(path, audio, metrics)
    np.save(path.with_suffix(".npy"), frames.numpy(), allow_pickle=False)
    takes[name] = dict(text=text, codes=torch.from_numpy(prepare_codes(frames.numpy()).astype(np.int64)))
    print(json.dumps(dict(case=name, **metrics)), flush=True)
    return dict(case=name, **metrics, wav_sha256=sha_file(path))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-expert-layers", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    identity = verify_assets(args.bundle, args.codec)
    torch.set_num_threads(args.threads)
    samples, source = reference_audio(args.reference)
    with CodecBridge(args.codec) as codec:
        reference = torch.from_numpy(codec.encode(samples, args.threads).astype(np.int64))
        started = time.perf_counter()
        model, bridge = load_model(args.bundle, threads=args.threads, gpu_layers=args.gpu_expert_layers)
        load_seconds = time.perf_counter() - started
        try:
            takes = {}
            records = [generate_case(model, codec, args, identity, reference, takes, case)
                       for case in CASES]
            report = dict(reference=source, reference_text=REFERENCE_TEXT,
                          load_seconds=load_seconds, cases=records)
            (args.output / "generation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        finally:
            bridge.close()


def main() -> None:
    from fish_drama3.watchdog import start_child_telemetry

    args = options()
    if os.environ.get("FISH_DRAMA3_WATCHDOG_ACTIVE") != "1":
        raise RuntimeError("Run this GPU example through python -m fish_drama3.watchdog")
    stop, thread = start_child_telemetry()
    try:
        run(args)
    finally:
        stop.set()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
