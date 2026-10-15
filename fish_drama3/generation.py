"""Bounded paragraph generation using the pinned model's audio history."""
from __future__ import annotations

import argparse
import re
from typing import Any, Callable

import numpy as np
import torch

from fish_drama3.codec_adapter import prepare_codes
from fish_drama3.hybrid import build_drama3_prompt, generate_drama3
from fish_drama3.text_chunks import split_text
from fish_speech.models.text2semantic.fish_omni.generate import validate_drama3_text


def generate_text(model: Any, args: argparse.Namespace, texts: list[str],
                  codes: list[torch.Tensor], *,
                  progress: Callable[[dict], None] | None = None) -> tuple[torch.Tensor, str, list[dict]]:
    validate_drama3_text(args.text, "text")
    chunks = split_text(args.text, args.chunk_length) if args.chunk_length else [args.text]
    history, outputs, segments = [], [], []
    speaker, reason, remaining = None, "length", args.max_frames
    for index, chunk in enumerate(chunks):
        if remaining < 1:
            reason = "length"
            break
        rendered = chunk
        if speaker and not re.match(r"<\|speaker:\d+\|>", chunk.lstrip()):
            rendered = speaker + " " + chunk
        prompt = build_drama3_prompt(rendered, model.tokenizer, model.config,
                                    texts, codes, history=history)
        segment_budget = frame_capacity(args, len(prompt), remaining)
        if progress:
            progress(segment_progress(args, chunks, index, segments, 0))
        frames, reason = generate_drama3(model, prompt, max_new_tokens=segment_budget,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
        if len(frames) == 0:
            raise RuntimeError("Model returned no acoustic frames for a text segment")
        outputs.append(frames)
        remaining -= len(frames)
        segments.append(dict(text=chunk, prompt_text=rendered, frames=len(frames),
                             finish_reason=reason, prompt_rows=len(prompt)))
        if progress:
            progress(segment_progress(args, chunks, index, segments[:-1], len(frames),
                                      complete=reason == "eos"))
        if reason != "eos":
            break
        previous = torch.from_numpy(prepare_codes(frames.numpy()).astype(np.int64))
        history = [(rendered, previous)]
        speakers = re.findall(r"<\|speaker:\d+\|>", rendered)
        speaker = speakers[-1] if speakers else speaker
    return torch.cat(outputs), reason, segments


def frame_capacity(args: argparse.Namespace, prompt_rows: int, remaining: int) -> int:
    limit = getattr(args, "max_context_rows", None)
    budget = remaining
    if args.chunk_length and limit is not None:
        budget = min(remaining, 8192, limit - prompt_rows)
    if budget < 1 or (limit is not None and prompt_rows + budget > limit):
        raise ValueError("Prompt plus audio frame budget exceeds 10240 context rows; "
                         "use smart segments, shorten the script/reference, or reduce "
                         "the one-shot frame budget. No speech generation has started "
                         "for this passage.")
    return budget


def segment_progress(args: argparse.Namespace, chunks: list[str], index: int,
                     previous: list[dict], current_frames: int,
                     complete: bool = False) -> dict:
    from fish_drama3.web_planning import estimated_frames
    completed = sum(segment["frames"] for segment in previous)
    characters = sum(len(segment["text"]) for segment in previous)
    return dict(frames=completed + current_frames, budget=args.max_frames,
                completed_frames=completed, segment_frames=current_frames,
                segment_index=index + 1, segment_count=len(chunks),
                segment_characters=len(chunks[index]),
                completed_characters=characters + (len(chunks[index]) if complete else 0),
                total_characters=len(args.text), segment_complete=complete,
                segment_estimated_frames=estimated_frames(chunks[index]),
                estimated_frames=sum(estimated_frames(chunk) for chunk in chunks))
