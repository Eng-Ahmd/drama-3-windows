"""Validated inputs shared by the local speech studio and its worker."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

DEFAULTS = dict(seed=123, temperature=0.7, top_p=0.7, top_k=30,
                max_frames=512, chunk_length=200, allow_partial=False,
                output_format="wav", mp3_bitrate=192, generation_mode="single")


def integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be {minimum}..{maximum}")
    return value


def model_options(data: dict) -> dict:
    bundle = Path(str(data.get("bundle", ""))).expanduser().resolve()
    codec = Path(str(data.get("codec", ""))).expanduser().resolve()
    if not (bundle / "model.gguf").is_file() or not (bundle / "manifest.json").is_file():
        raise ValueError("Choose a model bundle folder containing model.gguf and manifest.json")
    if not codec.is_file() or not (codec.parent / "manifest.json").is_file():
        raise ValueError("Choose codec.gguf with its adjacent manifest.json")
    return dict(bundle=str(bundle), codec=str(codec),
                gpu_expert_layers=integer(data.get("gpu_expert_layers", 12),
                                          "GPU expert layers", 0, 40),
                threads=integer(data.get("threads", 8), "CPU threads", 1, 256),
                **placement_options(data))


def placement_options(data: dict) -> dict:
    mode = data.get("placement_mode", "auto")
    if mode not in ("auto", "manual"):
        raise ValueError("Placement mode must be auto or manual")
    percent = data.get("headroom_percent", 3)
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        raise ValueError("VRAM headroom must be a number")
    if not math.isfinite(percent) or not 1 <= percent <= 25:
        raise ValueError("VRAM headroom must be 1..25 percent")
    return dict(placement_mode=mode, headroom_percent=percent)


def generation_options(data: dict, mp3_available: bool) -> dict:
    options = {key: data.get(key, value) for key, value in DEFAULTS.items()}
    for key, low, high in [("seed", 0, 2**32 - 1), ("top_k", 1, 2**31 - 1),
                           ("max_frames", 1, 8192), ("chunk_length", 0, 2000)]:
        options[key] = integer(options[key], key, low, high)
    if 0 < options["chunk_length"] < 32:
        raise ValueError("Chunk length must be 0 or 32..2000")
    if options["generation_mode"] not in ("single", "chunked"):
        raise ValueError("Generation mode must be single or chunked")
    if options["generation_mode"] == "single":
        options["chunk_length"] = 0
    for key in ("temperature", "top_p"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if options["top_p"] > 1:
        raise ValueError("Top-p must be at most 1")
    if not isinstance(options["allow_partial"], bool):
        raise ValueError("allow_partial must be true or false")
    if options["output_format"] not in ("wav", "mp3"):
        raise ValueError("Output format must be wav or mp3")
    if options["mp3_bitrate"] not in (128, 192, 256, 320):
        raise ValueError("MP3 bitrate must be 128, 192, 256 or 320")
    if options["output_format"] == "mp3" and not mp3_available:
        raise ValueError("Install FFmpeg with libmp3lame on PATH for MP3 export, or choose WAV")
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip() or len(text) > 20000:
        raise ValueError("Enter 1..20000 characters of speech text")
    options["text"] = text
    reference_text = data.get("reference_text", "")
    if not isinstance(reference_text, str) or len(reference_text) > 10000:
        raise ValueError("Reference transcript must be text, at most 10000 characters")
    options["reference_text"] = reference_text.strip()
    return options
