"""Lossless splitting of long text into bounded inference chunks."""

from __future__ import annotations

import re


_ATOMIC = re.compile(r"<\|[^>]*\|>|\[[^\]]*\]")
_SENTENCE_ENDS = ".?!。！？"


def _legal_cut_positions(text: str) -> set[int]:
    """Return character boundaries that do not divide recognized tags."""
    cuts = set(range(1, len(text)))
    for match in _ATOMIC.finditer(text):
        if match.end() - match.start() > 1:
            cuts.difference_update(range(match.start() + 1, match.end()))
    return cuts


def _preferred_cut(text: str, start: int, end: int, legal: set[int]) -> int:
    """Choose sentence, whitespace, or raw boundary in the current window."""
    for index in range(end - 1, start, -1):
        boundary = index + 1
        if _eligible_cut(text, start, boundary, legal) and text[index] in _SENTENCE_ENDS:
            if text[index] != "." or not (
                index > 0 and index + 1 < len(text)
                and text[index - 1].isdigit() and text[index + 1].isdigit()
            ):
                return boundary
    for index in range(end - 1, start, -1):
        if text[index].isspace() and _eligible_cut(text, start, index + 1, legal):
            return index + 1
    for boundary in range(end, start, -1):
        if _eligible_cut(text, start, boundary, legal):
            return boundary
    raise ValueError("text cannot be split without dividing an atomic tag or empty chunk")


def _eligible_cut(text: str, start: int, boundary: int, legal: set[int]) -> bool:
    return (boundary in legal and bool(text[start:boundary].strip())
            and bool(text[boundary:].strip()))


def split_text(text: str, limit: int = 200) -> list[str]:
    """Split text losslessly into non-whitespace chunks no longer than limit."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 32 <= limit <= 2000:
        raise ValueError("limit must be an integer from 32 through 2000")
    if not text or not text.strip():
        raise ValueError("text must contain non-whitespace characters")
    if len(text) > limit * 32:
        raise ValueError("text exceeds the maximum of 32 chunks")
    if any(match.end() - match.start() > limit for match in _ATOMIC.finditer(text)):
        raise ValueError("an atomic tag is longer than the chunk limit")

    legal = _legal_cut_positions(text)
    chunks: list[str] = []
    start = 0
    while len(text) - start > limit:
        cut = _preferred_cut(text, start, start + limit, legal)
        chunks.append(text[start:cut])
        start = cut
    chunks.append(text[start:])

    if len(chunks) > 32 or any(len(chunk) > limit for chunk in chunks):
        raise ValueError("text cannot be split into at most 32 bounded chunks")
    if any(not chunk.strip() for chunk in chunks):
        raise ValueError("text cannot be split without an empty chunk")
    return chunks
