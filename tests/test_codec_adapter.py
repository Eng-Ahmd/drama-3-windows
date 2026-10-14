"""Codec adapter arithmetic, exact axes, metadata, and fail-closed inputs."""
import json
import struct

import numpy as np
import pytest
import torch

from fish_drama3.codec_adapter import (
    codec_metadata, fold_legacy, plan_codec, prepare_codes, read_header,
    validate_payload, verify_source, write_payloads,
)
from fish_drama3.gguf_stream import StreamingGGUFWriter


def valid_rows() -> np.ndarray:
    codes = np.arange(30, dtype=np.int64).reshape(3, 10)
    return np.column_stack((codes[:, 0] + 248099, codes))


def test_codes_are_codebook_major_without_main_token() -> None:
    rows = valid_rows()
    result = prepare_codes(rows)
    assert result.dtype == np.int32 and result.flags.c_contiguous
    np.testing.assert_array_equal(result, rows[:, 1:].T)


@pytest.mark.parametrize("kind", ["shape", "empty", "float", "negative", "semantic", "residual", "main"])
def test_invalid_codes_rejected(kind: str) -> None:
    rows = valid_rows()
    if kind == "shape":
        rows = rows[:, :10]
    elif kind == "empty":
        rows = rows[:0]
    elif kind == "float":
        rows = rows.astype(np.float32)
    elif kind == "main":
        rows[0, 0] += 1
    else:
        column = 1 if kind == "semantic" else 2
        rows[0, column] = dict(negative=-1, semantic=4096, residual=1024)[kind]
    with pytest.raises(ValueError):
        prepare_codes(rows)


def test_fold_matches_torch_legacy_weight_norm() -> None:
    vector = np.arange(1, 25, dtype=np.float32).reshape(3, 8, 1)
    gain = np.array([0.25, 2, 7], dtype=np.float32).reshape(3, 1, 1)
    expected = torch._weight_norm(torch.tensor(vector), torch.tensor(gain), 0)
    np.testing.assert_allclose(fold_legacy(gain, vector), expected.numpy(), rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("bad", ["zero", "nan", "geometry"])
def test_bad_folds_rejected(bad: str) -> None:
    vector = np.ones((2, 3, 1), dtype=np.float32)
    gain = np.ones((2, 1, 1), dtype=np.float32)
    if bad == "zero":
        vector[:] = 0
    elif bad == "nan":
        vector[0, 0, 0] = np.nan
    else:
        gain = gain.reshape(2, 1)
    with pytest.raises(ValueError):
        fold_legacy(gain, vector)


def test_plan_does_not_refold_existing_weight() -> None:
    header = {
        "conv.weight": dict(dtype="F32", shape=[2, 3, 7]),
        "vq.weight_g": dict(dtype="F32", shape=[2, 1, 1]),
        "vq.weight_v": dict(dtype="F32", shape=[2, 3, 1]),
        "encoder.causal_mask": dict(dtype="BOOL", shape=[10, 10]),
        "encoder.freqs_cis": dict(dtype="BF16", shape=[10, 32, 2]),
    }
    outputs, bases, omitted = plan_codec(header)
    assert set(outputs) == {"conv.weight", "vq.weight", "encoder.freqs_cis"}
    assert bases == ["vq"] and len(omitted) == 1
    del header["vq.weight_v"]
    with pytest.raises(ValueError, match="Incomplete"):
        plan_codec(header)


def test_bf16_allowed_only_for_rope_tables() -> None:
    header = {"encoder.freqs_cis": dict(dtype="BF16", shape=[4, 32, 2])}
    outputs, _, _ = plan_codec(header)
    assert outputs == header
    with pytest.raises(ValueError, match="F32 codec learned"):
        plan_codec({"conv.weight": dict(dtype="BF16", shape=[2, 3, 7])})
    with pytest.raises(ValueError, match="BF16 codec RoPE"):
        plan_codec({"encoder.freqs_cis": dict(dtype="F32", shape=[4, 32, 2])})


def test_source_requires_sibling_manifest_identity(tmp_path) -> None:
    source = tmp_path / "fixture.bin"
    source.write_bytes(b"fixture")
    digest = verify_source(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(codec=dict(file=source.name, sha256=digest))))
    assert verify_source(source) == digest
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="sibling bundle manifest"):
        verify_source(source)


def test_bf16_rope_stream_preserves_source_bits(tmp_path) -> None:
    gguf = pytest.importorskip("gguf")
    bits = np.arange(256, dtype=np.uint16).reshape(4, 32, 2) + 0x3f00
    source = tmp_path / "rope.raw"
    source.write_bytes(bits.tobytes())
    item = dict(dtype="BF16", shape=list(bits.shape), data_offsets=[0, bits.nbytes])
    writer = StreamingGGUFWriter(tmp_path / "rope.gguf", arch="fish-speech-codec")
    writer.add_tensor("encoder.freqs_cis", item["shape"], 30, bits.nbytes, source)
    write_payloads(writer, source, {"encoder.freqs_cis": item}, 0,
                   {"encoder.freqs_cis": item}, [])
    writer.finish()
    reader = gguf.GGUFReader(str(tmp_path / "rope.gguf"))
    tensor = reader.tensors[0]
    assert tensor.tensor_type == 30
    np.testing.assert_array_equal(tensor.shape, [2, 32, 4])
    assert tensor.data.tobytes() == bits.tobytes()
    source.write_bytes(np.full(bits.shape, 0x7f80, dtype=np.uint16).tobytes())
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_payload(source, 0, item)


def test_stream_reload_axes_and_legacy_folding(tmp_path) -> None:
    gguf = pytest.importorskip("gguf")
    values = {"conv.weight": np.arange(42, dtype=np.float32).reshape(2, 3, 7),
              "vq.weight_g": np.array([2, 3], dtype=np.float32).reshape(2, 1, 1),
              "vq.weight_v": np.arange(1, 7, dtype=np.float32).reshape(2, 3, 1)}
    header, payload, offset = {}, bytearray(), 0
    for name, value in values.items():
        header[name] = dict(dtype="F32", shape=list(value.shape),
                            data_offsets=[offset, offset + value.nbytes])
        payload.extend(value.tobytes())
        offset += value.nbytes
    encoded = json.dumps(header).encode()
    source = tmp_path / "codec.safetensors"
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    header, start = read_header(source)
    outputs, bases, _ = plan_codec(header)
    writer = StreamingGGUFWriter(tmp_path / "codec.gguf", arch="fish-speech-codec")
    for key, value_type, value in codec_metadata():
        writer.add_kv(key, value_type, value)
    for name, item in outputs.items():
        writer.add_tensor(name, item["shape"], 0, np.prod(item["shape"]) * 4,
                          source, start + item["data_offsets"][0])
    folds = write_payloads(writer, source, header, start, outputs, bases)
    writer.finish()
    reader = gguf.GGUFReader(str(tmp_path / "codec.gguf"))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    np.testing.assert_array_equal(tensors["conv.weight"].shape, [7, 3, 2])
    np.testing.assert_array_equal(tensors["conv.weight"].data, values["conv.weight"])
    np.testing.assert_allclose(tensors["vq.weight"].data,
        fold_legacy(values["vq.weight_g"], values["vq.weight_v"]), rtol=0, atol=0)
    assert len(folds) == 1
    assert reader.fields["fish_speech.codec.transformer.n_local_heads"].contents() == -1
    assert reader.fields["fish_speech.codec.rvq_transformer.window_size"].contents() == 128
    assert reader.fields["fish_speech.codec.encoder_rates"].contents() == [2, 4, 8, 8]
    # Original DAC derives frame_length=hop_length*product(quantizer factors).
    rates = reader.fields["fish_speech.codec.encoder_rates"].contents()
    factors = reader.fields["fish_speech.codec.quantizer_downsample_factor"].contents()
    assert reader.fields["fish_speech.codec.frame_length"].contents() == np.prod(rates) * np.prod(factors) == 2048
