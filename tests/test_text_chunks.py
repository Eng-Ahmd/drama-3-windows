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
    assert chunks[0].rstrip().endswith(".")


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
def test_whitespace_edges_do_not_force_a_word_cut(text: str) -> None:
    with pytest.raises(ValueError, match="raise the passage limit"):
        split_text(text, 32)


@pytest.mark.parametrize("text", ["x" * 33, "prefix " + "unbreakable" * 5])
def test_indivisible_words_request_larger_limit(text: str) -> None:
    with pytest.raises(ValueError, match="raise the passage limit"):
        split_text(text, 32)


@pytest.mark.parametrize("text", [
    "Extraordinary words remain intact, even near passage boundaries. " * 3,
    "The measurement is 1234567890.1234567890 units. " * 3,
    "你好世界こんにちはカタカナ中文测试" * 8,
    "汉字か\u3099きくけこ [laughs] <|speaker:0|> hello there! " * 4,
    "Cafe\u0301 and nai\u0308ve words stay intact. " * 5,
])
def test_smart_boundaries_preserve_words_and_unicode(text: str) -> None:
    import unicodedata

    chunks = assert_lossless_chunks(text, 40)
    offset = 0
    for chunk in chunks[:-1]:
        offset += len(chunk)
        before, after = text[offset - 1], text[offset]
        assert not unicodedata.category(after).startswith("M")
        if before.isascii() and after.isascii():
            assert not (before.isalnum() and after.isalnum())
    assert all("1234567890.1234567890" in chunk
               for chunk in chunks if "1234567890" in chunk)


def test_decimal_is_never_a_preferred_sentence_cut() -> None:
    chunks = assert_lossless_chunks("Read 1234567890.1234567890 exactly, then continue talking. " * 2, 32)
    assert sum("1234567890.1234567890" in chunk for chunk in chunks) == 2


def test_sentence_pause_stays_with_previous_passage() -> None:
    chunks = assert_lossless_chunks("Hi. " + "x" * 32, 32)
    assert chunks == ["Hi. ", "x" * 32]
@pytest.mark.parametrize("token", ["a" * 25 + "." + "b" * 25,
                                    "example.domain.with.a.very.long.filename.txt"])
def test_dotted_tokens_are_not_split_as_sentences(token):
    with pytest.raises(ValueError, match="without splitting"):
        split_text(token, 32)
