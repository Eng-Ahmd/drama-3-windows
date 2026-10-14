"""Bounded paragraph generation using the pinned model's audio history."""
from __future__ import annotations

import argparse
import re
from typing import Any

import numpy as np
import torch

from fish_drama3.codec_adapter import prepare_codes
from fish_drama3.hybrid import build_drama3_prompt, generate_drama3
from fish_drama3.text_chunks import split_text
from fish_speech.models.text2semantic.fish_omni.generate import validate_drama3_text


def generate_text(model: Any, args: argparse.Namespace, texts: list[str],
                  codes: list[torch.Tensor]) -> tuple[torch.Tensor, str, list[dict]]:
    validate_drama3_text(args.text, "text")
    chunks = split_text(args.text, args.chunk_length) if args.chunk_length else [args.text]
    history, outputs, segments = [], [], []
    speaker, reason, remaining = None, "length", args.max_frames
    for chunk in chunks:
        if remaining < 1:
            reason = "length"
            break
        rendered = chunk
        if speaker and not re.match(r"<\|speaker:\d+\|>", chunk.lstrip()):
            rendered = speaker + " " + chunk
        prompt = build_drama3_prompt(rendered, model.tokenizer, model.config,
                                    texts, codes, history=history)
        frames, reason = generate_drama3(model, prompt, max_new_tokens=remaining,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
        if len(frames) == 0:
            raise RuntimeError("Model returned no acoustic frames for a text segment")
        outputs.append(frames)
        remaining -= len(frames)
        segments.append(dict(text=chunk, prompt_text=rendered, frames=len(frames),
                             finish_reason=reason, prompt_rows=len(prompt)))
        if reason != "eos":
            break
        previous = torch.from_numpy(prepare_codes(frames.numpy()).astype(np.int64))
        history = [(rendered, previous)]
        speakers = re.findall(r"<\|speaker:\d+\|>", rendered)
        speaker = speakers[-1] if speakers else speaker
    return torch.cat(outputs), reason, segments
