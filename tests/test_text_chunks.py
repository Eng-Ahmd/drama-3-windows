import pytest

from fish_drama3.text_chunks import split_text


def assert_lossless_chunks(text: str, limit: int) -> list[str]:
    chunks = split_text(text, limit)
    assert "".join(chunks) == text
    assert all(chunk.strip() and len(chunk) <= limit for chunk in chunks)
    assert len(chunks) <= 32
    return chunks


def test_english_sentences_split_at_sentence_boundaries() -> None:
    text = "First sentence. " + "Second sentence. " + "Third sentence." + "x" * 170
    chunks = assert_lossless_chunks(text, 200)
    assert len(chunks) == 2
    assert chunks[0].endswith(".")


def test_cjk_and_decimal_text_preserve_every_character() -> None:
    text = ("你好世界。" * 10) + "The time is 9.30 exactly. " + ("再见！" * 10)
    chunks = assert_lossless_chunks(text, 32)
    assert "9.30" in "".join(chunks)


def test_atomic_tags_and_speaker_labels_are_never_split() -> None:
    text = "[SPEAKER_ID_0] " + "a" * 28 + " <|audio|> " + "b" * 30
    chunks = assert_lossless_chunks(text, 32)
    for tag in ("[SPEAKER_ID_0]", "<|audio|>"):
        assert sum(tag in chunk for chunk in chunks) == 1


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_rejects_empty_or_whitespace_text(text: str) -> None:
    with pytest.raises(ValueError):
        split_text(text)


def test_rejects_oversized_tag_and_more_than_32_chunks() -> None:
    with pytest.raises(ValueError, match="atomic tag"):
        split_text("<|" + "x" * 31 + "|>", 32)
    with pytest.raises(ValueError, match="32 chunks"):
        split_text("x" * (32 * 32 + 1), 32)


def test_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        split_text("hello", 31)


@pytest.mark.parametrize("text", [" " * 30 + "hello", "x" * 32 + "\n"])
def test_whitespace_edges_do_not_create_empty_chunks(text: str) -> None:
    assert_lossless_chunks(text, 32)
