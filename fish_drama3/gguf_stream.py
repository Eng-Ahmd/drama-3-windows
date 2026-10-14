#!/usr/bin/env python3
"""Streaming GGUF v3 writer: precomputed layout, chunked data streaming.

gguf-py's writer retains every supplied array (120 GiB for full expert
banks), so full assembly needs this bounded alternative: register all
tensor metas upfront (names/shapes/types/sizes known, no payloads),
compute the exact layout, write header + KV + infos, then stream each
payload from disk in chunks with per-tensor hashes. Peak RAM stays flat.

finalize(): all-at-once convenience (bounded runs).
begin()/stream_one()/finish(): incremental mode for full runs —
  spool one layer, stream its banks, delete the spool, repeat.
Validated by reopening with gguf-py's independent GGUFReader.
"""
from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterable
from pathlib import Path

MAGIC = 0x46554747
VERSION = 3
ALIGN = 32

T_STRING, T_UINT32, T_BOOL = 8, 4, 7
T_INT32, T_FLOAT32, T_ARRAY = 5, 6, 9
GGML_F32 = 0
MAX_RESOURCE_BYTES = 64_000_000


def _pack_str(b: bytearray, s: str) -> None:
    e = s.encode("utf-8")
    b += struct.pack("<Q", len(e))
    b += e


class StreamingGGUFWriter:
    def __init__(self, path: Path, arch: str = "fish_drama3", alignment: int = ALIGN):
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Alignment must be a positive power of two")
        self.path = Path(path)
        self.arch = arch
        self.alignment = alignment
        self.kv: list[tuple[str, int, object]] = [("general.architecture", T_STRING, arch)]
        self.kv.append(("general.alignment", T_UINT32, alignment))
        self.tensors: list[dict] = []
        self._handle = None

    def add_kv(self, key: str, vtype: int, value: object) -> None:
        if self._handle is not None or any(k == key for k, _, _ in self.kv):
            raise ValueError(f"Late or duplicate metadata: {key}")
        self.kv.append((key, vtype, value))

    def add_resource(self, key: str, value: str, max_bytes: int = 8_000_000) -> None:
        """Keep each UTF-8 metadata string below the Hub parser's 10 MB limit."""
        if not 4 <= max_bytes <= 8_000_000:
            raise ValueError("Resource string limit must be 4..8000000 bytes")
        size = len(value.encode("utf-8"))
        if size > MAX_RESOURCE_BYTES:
            raise ValueError("Resource exceeds 64 MB")
        if size <= max_bytes:
            self.add_kv(key, T_STRING, value)
            return
        width = max_bytes // 4
        chunks = [value[start:start + width] for start in range(0, len(value), width)]
        if len(chunks) > 64:
            raise ValueError("Too many resource chunks")
        self.add_kv(key + ".chunk_count", T_UINT32, len(chunks))
        for index, chunk in enumerate(chunks):
            self.add_kv(f"{key}.chunk.{index:04d}", T_STRING, chunk)

    def add_tensor(self, name: str, logical_shape: list[int], ggml_type: int,
                   nbytes: int, src_path: Path, src_offset: int = 0) -> None:
        """Register a payload slice; nothing is read yet."""
        if self._handle is not None or any(t["name"] == name for t in self.tensors):
            raise ValueError(f"Late or duplicate tensor: {name}")
        if not logical_shape or any(d <= 0 for d in logical_shape) or nbytes <= 0:
            raise ValueError(f"Invalid tensor geometry: {name}")
        self.tensors.append({"name": name, "dims": list(reversed(logical_shape)),
                             "type": ggml_type, "nbytes": nbytes,
                             "src": Path(src_path), "off": src_offset})

    def _pack_kv(self, b: bytearray, vtype: int, value: object) -> None:
        b += struct.pack("<I", vtype)
        if vtype == T_STRING:
            _pack_str(b, value)  # type: ignore[arg-type]
        elif vtype == T_UINT32:
            b += struct.pack("<I", value)  # type: ignore
        elif vtype == T_BOOL:
            b += struct.pack("<?", value)  # type: ignore
        elif vtype == T_INT32:
            b += struct.pack("<i", value)  # type: ignore
        elif vtype == T_FLOAT32:
            b += struct.pack("<f", value)  # type: ignore
        elif vtype == T_ARRAY:
            subtype, values = value  # type: ignore
            formats = {T_UINT32: "I", T_INT32: "i", T_FLOAT32: "f"}
            if subtype not in formats:
                raise ValueError(f"unsupported array subtype {subtype}")
            b += struct.pack("<IQ", subtype, len(values))
            b += struct.pack("<" + formats[subtype] * len(values), *values)
        else:
            raise ValueError(f"unsupported KV type {vtype}")

    def _header_bytes(self) -> tuple[bytearray, int]:
        off = 0
        for t in self.tensors:
            t["offset"] = off
            off += (t["nbytes"] + self.alignment - 1) // self.alignment * self.alignment
        head = bytearray()
        head += struct.pack("<I", MAGIC)
        head += struct.pack("<I", VERSION)
        head += struct.pack("<Q", len(self.tensors))
        head += struct.pack("<Q", len(self.kv))
        for key, vtype, value in self.kv:
            _pack_str(head, key)
            self._pack_kv(head, vtype, value)
        for t in self.tensors:
            _pack_str(head, t["name"])
            head += struct.pack("<I", len(t["dims"]))
            for d in t["dims"]:
                head += struct.pack("<Q", d)
            head += struct.pack("<I", t["type"])
            head += struct.pack("<Q", t["offset"])
        data_start = (len(head) + self.alignment - 1) // self.alignment * self.alignment
        return head, data_start

    def begin(self) -> None:
        """Write header; returns nothing. Follow with stream_one() calls."""
        head, self._data_start = self._header_bytes()
        self._handle = open(self.path, "wb")
        self._handle.write(head)
        self._handle.write(b"\x00" * (self._data_start - len(head)))
        self._streamed: list[dict] = []

    def stream_one(self, name: str, chunk: int = 1 << 26) -> dict:
        """Stream one registered tensor by name; returns its manifest entry."""
        t = next(x for x in self.tensors if x["name"] == name)
        with open(t["src"], "rb") as src:
            src.seek(t["off"])
            return self.stream_payload(name, self._chunks(src, t["nbytes"], chunk))

    @staticmethod
    def _chunks(source, remaining: int, chunk: int) -> Iterable[bytes]:
        if chunk <= 0:
            raise ValueError("Chunk size must be positive")
        while remaining:
            data = source.read(min(chunk, remaining))
            if not data:
                raise EOFError("Short source tensor payload")
            remaining -= len(data)
            yield data

    def stream_payload(self, name: str, chunks: Iterable[bytes]) -> dict:
        """Write generated quantized chunks directly, without an intermediate file."""
        if self._handle is None:
            raise ValueError("begin() must precede payload writes")
        t = next(x for x in self.tensors if x["name"] == name)
        if any(s["tensor"] == name for s in self._streamed):
            raise ValueError(f"Duplicate payload: {name}")
        self._handle.seek(self._data_start + t["offset"])
        h = hashlib.sha256()
        count = 0
        for data in chunks:
            count += len(data)
            if count > t["nbytes"]:
                raise ValueError(f"Oversized payload: {name}")
            self._handle.write(data)
            h.update(data)
        if count != t["nbytes"]:
            raise ValueError(f"Short payload: {name}: {count}/{t['nbytes']}")
        pad = (self.alignment - t["nbytes"] % self.alignment) % self.alignment
        if pad:
            self._handle.write(b"\x00" * pad)
        rec = {"tensor": name, "dims_ggml": t["dims"], "type": t["type"],
               "bytes": t["nbytes"], "sha256": h.hexdigest()}
        self._streamed.append(rec)
        return rec

    def finish(self) -> dict:
        assert self._handle is not None, "begin() first"
        missing = [t["name"] for t in self.tensors
                   if t["name"] not in {s["tensor"] for s in self._streamed}]
        if missing:
            raise ValueError(f"unstreamed tensors: {missing[:5]}")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        total = self.path.stat().st_size
        return {"file": self.path.name, "bytes": total, "tensors": self._streamed}

    def abort(self) -> None:
        """Release the file on failure; callers retain/remove staging explicitly."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def finalize(self, chunk: int = 1 << 26) -> dict:
        self.begin()
        for t in self.tensors:
            self.stream_one(t["name"], chunk)
        return self.finish()


def read_resource_text(fields: dict, key: str) -> str:
    """Read legacy or chunked resource metadata without changing its content."""
    counter = key + ".chunk_count"
    if key in fields:
        if counter in fields:
            raise ValueError("Ambiguous resource encoding")
        value = fields[key].contents()
        if not isinstance(value, str):
            raise ValueError("Resource must be text")
        return value
    if counter not in fields:
        raise ValueError(f"Missing resource: {key}")
    count = fields[counter].contents()
    if not isinstance(count, int) or not 1 <= count <= 64:
        raise ValueError("Invalid resource chunk count")
    chunks = []
    total = 0
    for index in range(count):
        field = fields.get(f"{key}.chunk.{index:04d}")
        if field is None:
            raise ValueError("Missing resource chunk")
        value = field.contents()
        if not isinstance(value, str):
            raise ValueError("Resource chunk must be text")
        total += len(value.encode("utf-8"))
        if total > MAX_RESOURCE_BYTES:
            raise ValueError("Resource exceeds 64 MB")
        chunks.append(value)
    return "".join(chunks)
