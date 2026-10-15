"""Lossless passage splitting with software bounds and intact words/tags."""
from __future__ import annotations

import re
import unicodedata

_ATOMIC = re.compile(r"<\|[^>]*\|>|\[[^\]]*\]")
_SENTENCE_ENDS = ".?!。！？"


def _is_cjk(character: str) -> bool:
    value = ord(character)
    return (0x3400 <= value <= 0x9FFF or 0x20000 <= value <= 0x323AF
            or 0x3040 <= value <= 0x30FF or 0x31F0 <= value <= 0x31FF
            or 0xFF66 <= value <= 0xFF9D)


def _sentence_boundary(text: str, boundary: int) -> bool:
    previous = text[boundary - 1]
    if (boundary < len(text) and not text[boundary].isspace()
            and not _is_cjk(text[boundary])):
        return False
    return previous in _SENTENCE_ENDS and not (
        previous == "." and boundary > 1 and boundary < len(text)
        and text[boundary - 2].isdigit() and text[boundary].isdigit()
    )


def _legal_cut_positions(text: str) -> set[int]:
    """Keep words intact; allow CJK script boundaries without dividing marks."""
    cuts: set[int] = set()
    for boundary in range(1, len(text)):
        previous, following = text[boundary - 1], text[boundary]
        if (unicodedata.category(following).startswith("M")
                or previous == "\u200d" or following == "\u200d"
                or 0xFE00 <= ord(following) <= 0xFE0F
                or following in "\uff9e\uff9f"):
            continue
        if (previous.isspace() or following.isspace()
                or _sentence_boundary(text, boundary)
                or _is_cjk(previous) or _is_cjk(following)):
            cuts.add(boundary)
    for match in _ATOMIC.finditer(text):
        cuts.difference_update(range(match.start() + 1, match.end()))
    return cuts


def _preferred_cut(text: str, start: int, end: int, legal: set[int]) -> int:
    candidates = [boundary for boundary in range(end, start, -1)
                  if boundary in legal and text[start:boundary].strip()
                  and text[boundary:].strip()]
    for boundary in candidates:
        if _sentence_boundary(text, boundary):
            while boundary < end and text[boundary].isspace():
                boundary += 1
            return boundary
    for boundary in candidates:
        if text[boundary - 1].isspace():
            return boundary
    if candidates:
        return candidates[0]
    raise ValueError("A word or protected span cannot fit without splitting it; "
                     "raise the passage limit (software range 32–2000 characters) "
                     "or shorten the indivisible word/span.")


def split_text(text: str, limit: int = 200) -> list[str]:
    """Split losslessly; 32–2000 characters and 32 passages are software bounds."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 32 <= limit <= 2000:
        raise ValueError("limit must be an integer from 32 through 2000 (software safety bound)")
    if not text or not text.strip():
        raise ValueError("text must contain non-whitespace characters")
    if len(text) > limit * 32:
        raise ValueError("text exceeds the maximum of 32 chunks")
    if any(match.end() - match.start() > limit for match in _ATOMIC.finditer(text)):
        raise ValueError("an atomic tag is longer than the chunk limit; raise the passage limit")
    legal = _legal_cut_positions(text)
    chunks: list[str] = []
    start = 0
    while len(text) - start > limit:
        cut = _preferred_cut(text, start, start + limit, legal)
        chunks.append(text[start:cut])
        start = cut
        if len(chunks) >= 32:
            raise ValueError("text cannot be split into at most 32 bounded chunks")
    chunks.append(text[start:])
    if any(not chunk.strip() for chunk in chunks):
        raise ValueError("text cannot be split without an empty chunk; raise the passage limit")
    return chunks
