"""Independent decoding and boundary checks for the native quantizer."""
import gguf
import numpy as np
import pytest

from fish_drama3 import native_quant


@pytest.mark.parametrize("kind,relative_error", [(8, 0.01), (12, 0.10)])
def test_quantization_roundtrip(kind: int, relative_error: float) -> None:
    source = np.random.default_rng(2718).normal(size=(3, 1024)).astype(np.float32)
    encoded = native_quant.quantize(source, kind)
    block, block_bytes = native_quant.TYPE_SIZES[kind]
    assert len(encoded) == source.size // block * block_bytes
    assert encoded == native_quant.quantize(source, kind)
    decoded = gguf.dequantize(
        np.frombuffer(encoded, dtype=np.uint8), gguf.GGMLQuantizationType(kind)
    ).reshape(source.shape)
    assert np.isfinite(decoded).all()
    error = np.linalg.norm(decoded - source) / np.linalg.norm(source)
    assert error < relative_error
    # Separate row calls must preserve exactly the same serialized row order.
    assert encoded == b"".join(native_quant.quantize(row, kind) for row in source)


@pytest.mark.parametrize("kind", [8, 12])
def test_zero_and_strided_input(kind: int) -> None:
    source = np.zeros((2, 512), dtype=np.float32)[:, ::2]
    encoded = native_quant.quantize(source, kind)
    decoded = gguf.dequantize(
        np.frombuffer(encoded, dtype=np.uint8), gguf.GGMLQuantizationType(kind)
    )
    assert np.array_equal(decoded, np.zeros(source.size))


@pytest.mark.parametrize("shape", [(0, 256), (2, 257), (2, 2, 256)])
def test_bad_row_geometry(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        native_quant.quantize(np.zeros(shape, dtype=np.float32), 12)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejection(value: float) -> None:
    source = np.zeros(256, dtype=np.float32)
    source[0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        native_quant.quantize(source, 12)


def test_dtype_and_type_validation() -> None:
    with pytest.raises(TypeError):
        native_quant.quantize(np.zeros(256, dtype=np.float64), 12)
    with pytest.raises(ValueError, match="Unsupported"):
        native_quant.quantize(np.zeros(256, dtype=np.float32), 99)


def test_bounded_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_quant, "MAX_INPUT_BYTES", 1024)
    with pytest.raises(ValueError, match="stream smaller"):
        native_quant.quantize(np.zeros((2, 256), dtype=np.float32), 12)
