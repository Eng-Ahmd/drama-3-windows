"""Cheap, explicitly approximate speech budgeting before model inference."""
from __future__ import annotations

import math
import re

from fish_drama3.text_chunks import split_text

FRAME_SECONDS = 2048 / 44100
SINGLE_MAX_FRAMES = 8192
CHUNKED_MAX_FRAMES = 65536
_TAGS = re.compile(r"<\|[^>]*\|>|\[[^\]]*\]")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def speech_units(text: str) -> int:
    """Approximate spoken words; count unspaced CJK characters separately."""
    plain = _TAGS.sub(" ", text)
    cjk = len(_CJK.findall(plain))
    words = len(re.findall(r"\w+(?:['’\-]\w+)*", _CJK.sub(" ", plain)))
    return max(1, words + math.ceil(cjk / 2))


def estimated_frames(text: str, words_per_minute: int = 160) -> int:
    return max(1, math.ceil(speech_units(text) * 60 / words_per_minute / FRAME_SECONDS))


def recommended_chunk_length(text: str) -> int:
    """Choose a real, valid segmentation, including the 32-passage bound."""
    for length in (200, 256, 384, 512, 768, 1000, 1500, 2000):
        try:
            split_text(text, length)
            return length
        except ValueError:
            continue
    return 2000  # The planner will explain an unsplittable word/span on selection.


def generation_plan(options: dict) -> dict:
    """Validate actual boundaries and describe budget risk without running a model."""
    text, mode = options["text"], options["generation_mode"]
    limit = options["chunk_length"]
    chunks = split_text(text, limit) if mode == "chunked" else [text]
    expected = sum(estimated_frames(chunk) for chunk in chunks)
    conservative = sum(estimated_frames(chunk, 110) for chunk in chunks)
    budget = options["max_frames"]
    warnings: list[str] = []
    risk = "low"
    if conservative > budget:
        risk = "high"
        warnings.append("The selected total frame budget may end before the script does. "
                        "Increase the budget or use smart segments with a larger total budget.")
    elif mode == "single" and speech_units(text) > 200:
        risk = "warning"
    if mode == "single" and speech_units(text) > 200:
        warnings.append("Long one-shot speech can repeat, pause or omit words even when it fits. "
                        "Smart segments with audio history are the more reliable alternative.")
    if mode == "chunked":
        warnings.append("The frame budget covers the entire recording, not each segment. "
                        "Each segment also has its own prompt/context capacity.")
    maximum = SINGLE_MAX_FRAMES if mode == "single" else CHUNKED_MAX_FRAMES
    recommended = min(maximum, math.ceil(conservative * 1.15 / 128) * 128)
    return dict(mode=mode, characters=len(text), words=speech_units(text),
                segments=len(chunks), estimated_frames=expected,
                estimated_seconds=expected * FRAME_SECONDS,
                conservative_seconds=conservative * FRAME_SECONDS, max_frames=budget,
                budget_seconds=budget * FRAME_SECONDS, risk=risk, warnings=warnings,
                recommended_mode="chunked" if risk != "low" else mode,
                recommended_chunk_length=recommended_chunk_length(text),
                recommended_max_frames=recommended,
                recommended_chunked_frames=min(CHUNKED_MAX_FRAMES,
                                               math.ceil(conservative * 1.15 / 128) * 128),
                estimate_note="Speech duration is estimated at 160 words/min; "
                              "budget advice allows 110 words/min plus 15%. Not a guarantee.",
                chunks=[dict(characters=len(chunk), words=speech_units(chunk),
                             estimated_frames=estimated_frames(chunk)) for chunk in chunks])
