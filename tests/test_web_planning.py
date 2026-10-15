"""CPU-only checks for speech-budget advice and generation-mode limits."""
from __future__ import annotations

from pathlib import Path

import pytest

from fish_drama3.text_chunks import split_text
from fish_drama3.web_config import generation_options
from fish_drama3.web_planning import generation_plan, speech_units


def podcast_text() -> str:
    path = Path(__file__).resolve().parents[1] / "examples/samples/podcast.txt"
    return path.read_text(encoding="utf-8")


def test_podcast_budget_warns_before_inference() -> None:
    plan = generation_plan(generation_options(
        dict(text=podcast_text(), max_frames=6144), False))
    assert plan["words"] == 592
    assert plan["segments"] == 1
    assert plan["risk"] == "high"
    assert plan["warnings"]
    assert plan["recommended_chunked_frames"] > 6144
    assert plan["estimated_seconds"] < plan["budget_seconds"]


def test_larger_podcast_budget_still_warns_about_one_shot_fidelity() -> None:
    plan = generation_plan(generation_options(
        dict(text=podcast_text(), max_frames=8192), False))
    assert plan["risk"] in ("low", "warning")
    assert any("omit" in warning for warning in plan["warnings"])


def test_speech_units_count_unspaced_cjk_and_ignore_tags() -> None:
    assert speech_units("你好世界") == 2
    assert speech_units("こんにちはカタカナ") == 5
    assert speech_units("hello 你好世界") == 3
    assert speech_units("<|speaker:0|> [laughs] hello there") == 2
    assert speech_units("don't re-enter") == 2


def test_planning_uses_actual_lossless_passage_boundaries() -> None:
    text = "First sentence stays whole. Another sentence follows. " * 6
    options = generation_options(dict(text=text, generation_mode="chunked",
                                     chunk_length=64, max_frames=8192), False)
    plan = generation_plan(options)
    chunks = split_text(text, 64)
    assert plan["segments"] == len(chunks)
    assert [item["characters"] for item in plan["chunks"]] == list(map(len, chunks))
    assert sum(item["characters"] for item in plan["chunks"]) == len(text)


def test_long_segmented_recording_recommends_more_than_single_pass_cap() -> None:
    options = generation_options(dict(text="word " * 2000,
                                     generation_mode="chunked", chunk_length=1000,
                                     max_frames=65536), False)
    plan = generation_plan(options)
    assert 8192 < plan["recommended_max_frames"] <= 65536
    assert plan["segments"] <= 32
    assert plan["budget_seconds"] > plan["estimated_seconds"]
    assert any("entire recording" in warning for warning in plan["warnings"])


def test_segmented_total_budget_limit_does_not_apply_to_single_pass() -> None:
    request = dict(text="Hello world", max_frames=65536, chunk_length=200)
    options = generation_options(dict(request, generation_mode="chunked"), False)
    assert options["max_frames"] == 65536
    with pytest.raises(ValueError, match="max_frames"):
        generation_options(dict(request, generation_mode="single"), False)
    with pytest.raises(ValueError, match="max_frames"):
        generation_options(
            dict(request, generation_mode="chunked", max_frames=65537), False)


def test_segmented_zero_passage_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="passage length"):
        generation_options(dict(text="Hello", generation_mode="chunked",
                                chunk_length=0), False)
def test_long_script_gets_a_valid_adaptive_passage_size():
    from fish_drama3.web_planning import recommended_chunk_length
    from fish_drama3.text_chunks import split_text

    text = "A sentence with enough words to narrate. " * 220
    length = recommended_chunk_length(text)
    assert length > 200
    chunks = split_text(text, length)
    assert len(chunks) <= 32 and "".join(chunks) == text
