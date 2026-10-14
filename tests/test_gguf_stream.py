"""Verify streamed GGUF with an independent reader and native Q4 payload."""
import hashlib
from pathlib import Path

import gguf
import numpy as np
import pytest

from fish_drama3.gguf_stream import StreamingGGUFWriter, read_resource_text
from fish_drama3.native_quant import Q4_K, quantize


def test_codec_metadata_types(tmp_path: Path) -> None:
    writer = StreamingGGUFWriter(tmp_path / "codec.gguf", "fish-speech-codec")
    writer.add_kv("signed", 5, -1)
    writer.add_kv("epsilon", 6, 1e-5)
    writer.add_kv("rates", 9, (4, [2, 4, 8, 8]))
    writer.begin()
    writer.finish()
    reader = gguf.GGUFReader(str(tmp_path / "codec.gguf"))
    assert reader.fields["signed"].contents() == -1
    assert reader.fields["epsilon"].contents() == pytest.approx(1e-5)
    assert reader.fields["rates"].contents() == [2, 4, 8, 8]


def test_quantized_payload_and_out_of_order_offsets(tmp_path: Path) -> None:
    source = np.linspace(-2, 2, 256, dtype=np.float32).reshape(1, 256)
    quantized = quantize(source, Q4_K)
    dense = np.array([13.0, 17.0, 19.0], dtype=np.float32)
    writer = StreamingGGUFWriter(tmp_path / "model.gguf")
    writer.add_tensor("experts", [1, 256], Q4_K, len(quantized), tmp_path)
    writer.add_tensor("norm", [3], 0, dense.nbytes, tmp_path)
    writer.begin()
    writer.stream_payload("norm", [dense.tobytes()])
    writer.stream_payload("experts", [quantized[:33], quantized[33:]])
    manifest = writer.finish()
    reader = gguf.GGUFReader(str(tmp_path / "model.gguf"))
    found = {tensor.name: tensor for tensor in reader.tensors}
    assert tuple(found["experts"].shape) == (256, 1)
    assert found["experts"].tensor_type == gguf.GGMLQuantizationType.Q4_K
    assert found["experts"].data.tobytes() == quantized
    np.testing.assert_array_equal(found["norm"].data.ravel(), dense)
    decoded = gguf.dequantize(found["experts"].data, gguf.GGMLQuantizationType.Q4_K)
    assert np.linalg.norm(decoded.reshape(source.shape) - source) / np.linalg.norm(source) < 0.1
    records = {item["tensor"]: item for item in manifest["tensors"]}
    assert records["experts"]["sha256"] == hashlib.sha256(quantized).hexdigest()
    assert found["norm"].data_offset - found["experts"].data_offset == 160


def writer_with_tensor(tmp_path: Path) -> StreamingGGUFWriter:
    writer = StreamingGGUFWriter(tmp_path / "model.gguf")
    writer.add_tensor("weight", [1], 0, 4, tmp_path)
    return writer


@pytest.mark.parametrize("payload,message", [(b"abc", "Short"), (b"abcde", "Oversized")])
def test_payload_length_failure_and_abort(tmp_path: Path, payload: bytes, message: str) -> None:
    writer = writer_with_tensor(tmp_path)
    writer.begin()
    with pytest.raises(ValueError, match=message):
        writer.stream_payload("weight", [payload])
    writer.abort()
    assert writer._handle is None
    with pytest.raises(ValueError, match="begin"):
        writer.stream_payload("weight", [b"abcd"])


def test_duplicate_registration_and_payload(tmp_path: Path) -> None:
    writer = writer_with_tensor(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        writer.add_tensor("weight", [1], 0, 4, tmp_path)
    writer.begin()
    writer.stream_payload("weight", [b"abcd"])
    with pytest.raises(ValueError, match="Duplicate payload"):
        writer.stream_payload("weight", [b"abcd"])
    writer.abort()


def test_incomplete_finish_fails(tmp_path: Path) -> None:
    writer = writer_with_tensor(tmp_path)
    writer.begin()
    with pytest.raises(ValueError, match="unstreamed"):
        writer.finish()
    writer.abort()


def test_short_source_file_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    writer = StreamingGGUFWriter(tmp_path / "model.gguf")
    writer.add_tensor("weight", [1], 0, 4, source)
    writer.begin()
    with pytest.raises(EOFError, match="Short"):
        writer.stream_one("weight", chunk=2)
    writer.abort()


@pytest.mark.parametrize("text", ["short", '{"text":"你好🦊 العربية"}' * 8], ids=["legacy", "chunked-unicode"])
def test_resource_text_preserves_unicode_and_legacy_encoding(tmp_path: Path, text: str) -> None:
    path = tmp_path / "resources.gguf"
    writer = StreamingGGUFWriter(path)
    writer.add_resource("resource", text, max_bytes=24)
    writer.begin()
    writer.finish()
    reader = gguf.GGUFReader(str(path))
    assert read_resource_text(reader.fields, "resource") == text
    for key, field in reader.fields.items():
        if key.startswith("resource.chunk."):
            assert len(field.contents().encode("utf-8")) <= 24
    if len(text.encode("utf-8")) <= 24:
        assert "resource" in reader.fields
    else:
        fields = dict(reader.fields)
        fields.pop("resource.chunk.0001")
        with pytest.raises(ValueError, match="Missing resource chunk"):
            read_resource_text(fields, "resource")
        fields = dict(reader.fields)
        fields["resource"] = reader.fields["resource.chunk.0000"]
        with pytest.raises(ValueError, match="Ambiguous"):
            read_resource_text(fields, "resource")


def test_resource_writer_rejects_oversize_before_emitting(tmp_path: Path, monkeypatch) -> None:
    import fish_drama3.gguf_stream as stream

    monkeypatch.setattr(stream, "MAX_RESOURCE_BYTES", 32)
    writer = StreamingGGUFWriter(tmp_path / "resource.gguf")
    previous = list(writer.kv)
    with pytest.raises(ValueError, match="exceeds"):
        writer.add_resource("resource", "🦊" * 9, max_bytes=16)
    assert writer.kv == previous
