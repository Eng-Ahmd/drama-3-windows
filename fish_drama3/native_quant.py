"""Bounded F32 quantization using the project's pinned CPU GGML library.

Input axes remain NumPy [rows, columns]; columns are GGML ne[0]. Callers
stream complete rows, keeping each call below MAX_INPUT_BYTES.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np

Q8_0 = 8
Q4_K = 12
# GGML type -> (elements per block, serialized bytes per block).
TYPE_SIZES: dict[int, tuple[int, int]] = {Q8_0: (32, 34), Q4_K: (256, 144)}
MAX_INPUT_BYTES = 64 * 1024 * 1024
DLL_PATH = Path(__file__).resolve().parents[1] / "builds/s2-cpu/bin/ggml-base.dll"


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    """Bind the ABI declared in the pinned ggml/include/ggml.h."""
    if not DLL_PATH.is_file():
        raise FileNotFoundError(f"Build the pinned s2 CPU GGML library: {DLL_PATH}")
    try:
        library = ctypes.CDLL(str(DLL_PATH))
        library.ggml_blck_size.argtypes = [ctypes.c_int]
        library.ggml_blck_size.restype = ctypes.c_int64
        library.ggml_type_size.argtypes = [ctypes.c_int]
        library.ggml_type_size.restype = ctypes.c_size_t
        library.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        library.ggml_quantize_chunk.restype = ctypes.c_size_t
        library.ggml_validate_row_data.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
        ]
        library.ggml_validate_row_data.restype = ctypes.c_bool
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"Cannot bind pinned GGML quantizer: {DLL_PATH}") from error
    for kind, (block, size) in TYPE_SIZES.items():
        actual = (library.ggml_blck_size(kind), library.ggml_type_size(kind))
        if actual != (block, size):
            raise RuntimeError(f"GGML ABI mismatch for type {kind}: {actual}")
    return library


def _validate(array: np.ndarray, ggml_type: int) -> tuple[int, int, int]:
    if isinstance(ggml_type, bool) or ggml_type not in TYPE_SIZES:
        raise ValueError(f"Unsupported GGML quantization type: {ggml_type}")
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype("float32"):
        raise TypeError("Quantization requires a native-endian float32 NumPy array")
    if array.ndim not in (1, 2) or array.size == 0:
        raise ValueError("Input must contain one or more complete 1D/2D rows")
    if array.nbytes > MAX_INPUT_BYTES:
        raise ValueError("Input exceeds 64 MiB; stream smaller complete-row chunks")
    columns = array.shape[-1]
    block, block_bytes = TYPE_SIZES[ggml_type]
    if columns % block:
        raise ValueError(f"Row length {columns} must be divisible by {block}")
    if not np.isfinite(array).all():
        raise ValueError("Input contains nonfinite values")
    rows = array.size // columns
    return rows, columns, array.size // block * block_bytes


def quantize(array: np.ndarray, ggml_type: int) -> bytes:
    """Return deterministic GGML block bytes for bounded float32 rows.

    Q4_K and Q8_0 use GGML's reference quantizers without an importance
    matrix. Quantization error is assessed separately by component tests.
    """
    rows, columns, byte_count = _validate(array, ggml_type)
    source = np.ascontiguousarray(array)
    destination = ctypes.create_string_buffer(byte_count)
    written = _library().ggml_quantize_chunk(
        ggml_type, source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        destination, 0, rows, columns, None,
    )
    if written != byte_count:
        raise RuntimeError(f"GGML wrote {written} bytes; expected {byte_count}")
    if not _library().ggml_validate_row_data(ggml_type, destination, byte_count):
        raise ValueError("Quantized output has nonfinite scales; input range is unsafe")
    return destination.raw
