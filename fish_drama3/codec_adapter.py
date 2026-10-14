"""Adapt folded codec weights to the pinned native donor; parity is unverified."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

from fish_drama3.gguf_stream import StreamingGGUFWriter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/drama3-q4k-q8/codec.fp32.safetensors"
DESTINATION = ROOT / "models/drama3-codec-adapter-v3"
BRIDGE = ROOT / "builds/drama3-inspect/Release/drama3-codec-bridge.dll"
PREFIX = "fish_speech.codec."


def codec_metadata() -> list[tuple[str, int, object]]:
    """Pinned modded_dac_vq YAML, using the donor's metadata schema."""
    integers = dict(sample_rate=44100, hop_length=512, frame_length=2048,
                    encoder_dim=64, decoder_dim=1536, latent_dim=1024,
                    quantizer_input_dim=1024, quantizer_codebook_dim=8,
                    quantizer_residual_codebooks=9,
                    quantizer_residual_codebook_size=1024,
                    quantizer_semantic_codebook_size=4096)
    arrays = dict(encoder_rates=[2, 4, 8, 8], decoder_rates=[8, 8, 4, 2],
                  encoder_transformer_layers=[0, 0, 0, 4],
                  decoder_transformer_layers=[4, 0, 0, 0],
                  quantizer_downsample_factor=[2, 2])
    records = [(PREFIX + key, 4, value) for key, value in integers.items()]
    records += [(PREFIX + key, 9, (4, value)) for key, value in arrays.items()]
    records.append((PREFIX + "quantizer_type", 8,
                    "downsample_residual_vector_quantize"))
    for scope, block in (("transformer", 8192), ("rvq_transformer", 2048)):
        records += [(PREFIX + scope + ".block_size", 4, block),
                    (PREFIX + scope + ".n_local_heads", 5, -1),
                    (PREFIX + scope + ".head_dim", 4, 64),
                    (PREFIX + scope + ".rope_freq_base", 6, 10000.0),
                    (PREFIX + scope + ".layer_norm_rms_eps", 6, 1e-5)]
    for key, value in dict(window_size=128, n_layer=8, n_head=16, dim=1024,
                           feed_forward_length=3072).items():
        records.append((PREFIX + "rvq_transformer." + key, 4, value))
    return records


def read_header(source: Path) -> tuple[dict, int]:
    with source.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("Unreasonable safetensors header")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def plan_codec(header: dict) -> tuple[dict, list[str], list[str]]:
    omitted = [name for name in header if name.endswith("causal_mask")]
    bases = sorted(name[:-9] for name in header if name.endswith(".weight_g"))
    consumed = {base + suffix for base in bases for suffix in (".weight_g", ".weight_v")}
    if not consumed <= header.keys():
        raise ValueError("Incomplete legacy weight-norm pair")
    if any(name.endswith(".weight_v") and name not in consumed for name in header):
        raise ValueError("Orphan legacy weight-norm vector")
    outputs = {name: header[name] for name in header if name not in consumed | set(omitted)}
    for base in bases:
        name = base + ".weight"
        if name in outputs:
            raise ValueError("Duplicate folded weight: " + name)
        outputs[name] = header[base + ".weight_v"]
    for name, item in outputs.items():
        if name.endswith("freqs_cis"):
            if item["dtype"] != "BF16":
                raise ValueError("Expected BF16 codec RoPE table")
        elif item["dtype"] != "F32":
            raise ValueError("Expected F32 codec learned tensors")
    return outputs, bases, sorted(omitted)


def fold_legacy(gain: np.ndarray, vector: np.ndarray) -> np.ndarray:
    import torch

    if vector.ndim < 2 or gain.shape != (vector.shape[0],) + (1,) * (vector.ndim - 1):
        raise ValueError("Invalid legacy weight-norm geometry")
    if not np.isfinite(gain).all() or not np.isfinite(vector).all():
        raise ValueError("Nonfinite legacy weight-norm input")
    weight = torch.from_numpy(vector.copy())
    norm = torch.linalg.vector_norm(weight, dim=tuple(range(1, weight.ndim)), keepdim=True)
    result = (torch.from_numpy(gain.copy()) * weight / norm).numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite folded weight")
    return np.ascontiguousarray(result, dtype="<f4")


def tensor_array(source: Path, offset: int, item: dict) -> np.ndarray:
    start, end = item["data_offsets"]
    shape = tuple(item["shape"])
    if item["dtype"] != "F32" or end - start != np.prod(shape) * 4:
        raise ValueError("Invalid F32 payload geometry")
    return np.memmap(source, dtype="<f4", mode="r", offset=offset + start, shape=shape)


def validate_payload(source: Path, offset: int, item: dict) -> None:
    start, end = item["data_offsets"]
    element_bytes = 2 if item["dtype"] == "BF16" else 4
    if end - start != int(np.prod(item["shape"])) * element_bytes:
        raise ValueError("Invalid codec payload geometry")
    if item["dtype"] == "BF16":
        array = np.memmap(source, dtype="<u2", mode="r", offset=offset + start,
                          shape=tuple(item["shape"]))
        if np.any((array & 0x7f80) == 0x7f80):
            raise ValueError("Nonfinite codec RoPE table")
    else:
        flat = tensor_array(source, offset, item).reshape(-1)
        for begin in range(0, flat.size, 1 << 22):
            if not np.isfinite(flat[begin:begin + (1 << 22)]).all():
                raise ValueError("Nonfinite codec tensor")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 26), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: Path) -> str:
    digest = sha_file(source)
    manifest_path = source.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        codec = manifest.get("codec", {})
        if codec.get("file") != source.name or codec.get("sha256") != digest:
            raise ValueError("Codec source does not match sibling bundle manifest")
    elif source.resolve() == SOURCE.resolve():
        raise ValueError("Default codec source requires its bundle manifest")
    return digest


def fold_comparison(gain: np.ndarray, vector: np.ndarray, result: np.ndarray) -> float:
    import torch

    expected = torch._weight_norm(torch.from_numpy(vector.copy()),
                                  torch.from_numpy(gain.copy()), 0).numpy()
    error = float(np.max(np.abs(expected - result)))
    if not np.allclose(expected, result, rtol=2e-7, atol=1e-7):
        raise ValueError("Legacy weight-norm reference mismatch")
    return error


def omitted_ledger(source: Path, header: dict, offset: int, names: list[str]) -> list[dict]:
    records = []
    with source.open("rb") as handle:
        for name in names:
            item = header[name]
            start, end = item["data_offsets"]
            handle.seek(offset + start)
            digest = hashlib.sha256()
            for data in StreamingGGUFWriter._chunks(handle, end - start, 1 << 26):
                digest.update(data)
            records.append(dict(tensor=name, dtype=item["dtype"], shape=item["shape"],
                                bytes=end - start, sha256=digest.hexdigest(),
                                reason="Native donor regenerates causal masks"))
    return records


def verify_regenerated_buffers(source: Path, header: dict, offset: int,
                               names: list[str]) -> list[dict]:
    """Check source tables against pinned triangular mask and RoPE formulas."""
    import torch

    records = []
    for name in names:
        item = header[name]
        start = offset + item["data_offsets"][0]
        shape = tuple(item["shape"])
        mismatches = 0
        if name.endswith("causal_mask"):
            if item["dtype"] != "BOOL" or len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("Unexpected source mask geometry")
            actual = np.memmap(source, mode="r", dtype=np.bool_, offset=start, shape=shape)
            columns = np.arange(shape[1])[None, :]
            for begin in range(0, shape[0], 128):
                rows = np.arange(begin, min(begin + 128, shape[0]))[:, None]
                mismatches += int(np.count_nonzero(actual[begin:begin + 128] != (columns <= rows)))
        else:
            if item["dtype"] != "BF16" or len(shape) != 3 or shape[1:] != (32, 2):
                raise ValueError("Unexpected source RoPE geometry")
            actual = np.memmap(source, mode="r", dtype="<u2", offset=start, shape=shape)
            frequencies = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
            angles = torch.outer(torch.arange(shape[0]), frequencies)
            polar = torch.polar(torch.ones_like(angles), angles)
            expected = torch.stack([polar.real, polar.imag], dim=-1).to(torch.bfloat16)
            bits = expected.view(torch.int16).numpy().view(np.uint16)
            mismatches = int(np.count_nonzero(actual != bits))
        if mismatches:
            raise ValueError(f"Regenerated buffer mismatch: {name}: {mismatches}")
        records.append(dict(tensor=name, elements=int(np.prod(shape)), mismatches=0))
    return records


def write_payloads(writer: StreamingGGUFWriter, source: Path, header: dict,
                   offset: int, outputs: dict, bases: list[str]) -> list[dict]:
    folds = []
    writer.begin()
    try:
        for name in sorted(outputs):
            base = name[:-7] if name.endswith(".weight") else ""
            if base in bases:
                gain = tensor_array(source, offset, header[base + ".weight_g"])
                vector = tensor_array(source, offset, header[base + ".weight_v"])
                result = fold_legacy(gain, vector)
                record = writer.stream_payload(name, [result.tobytes()])
                folds.append(dict(base=base, dim=0, sha256=record["sha256"],
                                  torch_reference_max_abs=fold_comparison(gain, vector, result)))
            else:
                validate_payload(source, offset, outputs[name])
                writer.stream_one(name)
        return folds
    except BaseException:
        writer.abort()
        raise


def build_codec(source: Path = SOURCE, destination: Path = DESTINATION) -> Path:
    source_hash = verify_source(source)
    header, offset = read_header(source)
    outputs, bases, omitted = plan_codec(header)
    rope_count = sum(name.endswith("freqs_cis") for name in outputs)
    if (len(outputs), len(bases), len(omitted), rope_count) != (458, 20, 3, 3):
        raise ValueError("Codec contract requires 458 outputs, 20 folds, 3 masks, 3 RoPE tables")
    if destination.exists():
        raise FileExistsError("Codec bundle already exists: " + str(destination))
    staging = destination.with_name(destination.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    writer = StreamingGGUFWriter(staging / "codec.gguf", arch="fish-speech-codec")
    for key, value_type, value in codec_metadata():
        writer.add_kv(key, value_type, value)
    for name, item in sorted(outputs.items()):
        tensor_type, element_bytes = (30, 2) if item["dtype"] == "BF16" else (0, 4)
        writer.add_tensor(name, item["shape"], tensor_type,
                          int(np.prod(item["shape"])) * element_bytes,
                          source, offset + item["data_offsets"][0])
    folds = write_payloads(writer, source, header, offset, outputs, bases)
    artifact = writer.finish()
    manifest = dict(source=str(source), source_sha256=source_hash,
                    config_revision="1ff3c3ed2f80d3106a83909dd25b83129c3d816f",
                    folded_legacy=folds,
                    omitted_regenerated=omitted_ledger(source, header, offset, omitted),
                    regenerated_formula_checks=verify_regenerated_buffers(source, header, offset,
                        omitted + [name for name in outputs if name.endswith("freqs_cis")]),
                    artifact=artifact, sha256=sha_file(staging / "codec.gguf"),
                    limitation="Native donor reuse; independent waveform parity unverified")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.rename(staging, destination)
    return destination / "codec.gguf"


def verify_existing(source: Path = SOURCE, destination: Path = DESTINATION) -> dict:
    """Add measured verification to an existing bundle without rewriting it."""
    source_hash = verify_source(source)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source_hash:
        raise ValueError("Existing adapter source mismatch")
    if sha_file(destination / "codec.gguf") != manifest["sha256"]:
        raise ValueError("Existing adapter file hash mismatch")
    header, offset = read_header(source)
    _, bases, omitted = plan_codec(header)
    outputs = {item["tensor"]: item for item in manifest["artifact"]["tensors"]}
    folds = []
    for base in bases:
        gain = tensor_array(source, offset, header[base + ".weight_g"])
        vector = tensor_array(source, offset, header[base + ".weight_v"])
        result = fold_legacy(gain, vector)
        digest = hashlib.sha256(result.tobytes()).hexdigest()
        if digest != outputs[base + ".weight"]["sha256"]:
            raise ValueError("Existing adapter fold payload mismatch")
        folds.append(dict(base=base, dim=0, sha256=digest,
                          torch_reference_max_abs=fold_comparison(gain, vector, result)))
    report = dict(source_sha256=source_hash, folded_legacy=folds,
                  omitted_regenerated=omitted_ledger(source, header, offset, omitted),
                  regenerated_formula_checks=verify_regenerated_buffers(source, header, offset, omitted),
                  limitation="Independent waveform parity remains unverified")
    (destination / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def prepare_codes(rows: np.ndarray) -> np.ndarray:
    if rows.ndim != 2 or rows.shape[1] != 11 or rows.shape[0] < 1:
        raise ValueError("Expected nonempty [frames,11] rows")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Codes must have integer dtype")
    codes = rows[:, 1:]
    if np.any(codes < 0) or np.any(codes[:, 0] >= 4096) or np.any(codes[:, 1:] >= 1024):
        raise ValueError("Codec index outside semantic/residual codebook range")
    if np.any(rows[:, 0] != codes[:, 0] + 248099):
        raise ValueError("Main token disagrees with semantic code")
    return np.ascontiguousarray(codes.T, dtype=np.int32)


class CodecBridge:
    """Owned native handle; finish decodes before closing it."""

    def __init__(self, codec: Path, library: Path = BRIDGE) -> None:
        self.handle = None
        self._directory = os.add_dll_directory(str(library.resolve().parent))
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.lib.drama3_codec_open.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.drama3_codec_open.restype = ctypes.c_void_p
        self.lib.drama3_codec_close.argtypes = [ctypes.c_void_p]
        self.lib.drama3_codec_close.restype = None
        self.lib.drama3_codec_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.drama3_codec_decode.restype = ctypes.c_int
        self.lib.drama3_codec_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.drama3_codec_encode.restype = ctypes.c_int
        error = ctypes.create_string_buffer(2048)
        self.handle = self.lib.drama3_codec_open(os.fsencode(codec.resolve()), error, len(error))
        if not self.handle:
            self.close()
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))

    def encode(self, audio: np.ndarray, threads: int = 8) -> np.ndarray:
        if not self.handle or not 1 <= threads <= 256:
            raise ValueError("Closed codec or invalid thread count")
        if audio.ndim != 1 or not 1 <= len(audio) <= 8_388_608:
            raise ValueError("Audio must be nonempty mono 44100 Hz samples within codec limits")
        values = np.ascontiguousarray(audio, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite reference audio")
        capacity = ((len(values) + 2047) // 2048) * 10
        output = np.empty(capacity, dtype=np.int32)
        frames = ctypes.c_int()
        error = ctypes.create_string_buffer(2048)
        status = self.lib.drama3_codec_encode(self.handle, values.ctypes.data,
            len(values), threads, output.ctypes.data, capacity, ctypes.byref(frames),
            error, len(error))
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        if not 1 <= frames.value <= capacity // 10:
            raise RuntimeError("Invalid native encoder output count")
        codes = output[:frames.value * 10].reshape(10, frames.value).copy()
        if np.any(codes < 0) or np.any(codes[0] >= 4096) or np.any(codes[1:] >= 1024):
            raise RuntimeError("Native encoder returned invalid codes")
        return codes

    def decode(self, rows: np.ndarray, threads: int = 8) -> np.ndarray:
        codes = prepare_codes(rows)
        if not self.handle or not 1 <= threads <= 256 or codes.shape[1] > (2**31 - 1) // 2048:
            raise ValueError("Closed codec or invalid decode dimensions/threads")
        output = np.empty(codes.shape[1] * 2048, dtype=np.float32)
        count = ctypes.c_int()
        error = ctypes.create_string_buffer(2048)
        status = self.lib.drama3_codec_decode(self.handle, codes.ctypes.data,
            codes.shape[1], threads, output.ctypes.data, output.size,
            ctypes.byref(count), error, len(error))
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        if not 0 < count.value <= output.size or not np.isfinite(output[:count.value]).all():
            raise RuntimeError("Invalid native codec output")
        return output[:count.value].copy()

    def close(self) -> None:
        if self.handle:
            self.lib.drama3_codec_close(self.handle)
            self.handle = None
        if getattr(self, "_directory", None):
            self._directory.close()
            self._directory = None

    def __enter__(self) -> CodecBridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build")
    build.add_argument("--source", type=Path, default=SOURCE)
    build.add_argument("--destination", type=Path, default=DESTINATION)
    decode = subcommands.add_parser("decode")
    decode.add_argument("codes", type=Path)
    decode.add_argument("wave", type=Path)
    decode.add_argument("--codec", type=Path, default=DESTINATION / "codec.gguf")
    decode.add_argument("--threads", type=int, default=8)
    options = parser.parse_args()
    if options.command == "build":
        print(build_codec(options.source, options.destination))
    else:
        import soundfile
        rows = np.load(options.codes, allow_pickle=False)
        with CodecBridge(options.codec) as bridge:
            audio = bridge.decode(rows, options.threads)
        soundfile.write(options.wave, audio, 44100, subtype="FLOAT")
        print(f"Wrote {len(audio)} finite samples at 44100 Hz")


if __name__ == "__main__":
    main()
