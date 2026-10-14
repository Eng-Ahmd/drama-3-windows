"""Direct checkpoint -> Q4_K/Q8_0 GGUF; no full FP32 intermediates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from fish_drama3.convert import dest_name, fp8_block_reconstruct, sha_file, write_codec
from fish_drama3.gguf_stream import StreamingGGUFWriter, T_BOOL, T_STRING
from fish_drama3.native_quant import DLL_PATH, Q4_K, Q8_0, TYPE_SIZES, quantize

REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
RECIPE = "Q4_K-experts_Q8_0-backbone"
EXPERT = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
SOURCE_WIDTH = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shape: tuple[int, ...]
    storage: int
    sources: tuple[str, ...]

    @property
    def nbytes(self) -> int:
        block, width = TYPE_SIZES.get(self.storage, (1, 2 if self.storage == 30 else 4))
        if self.shape[-1] % block:
            raise ValueError(f"Unaligned rows: {self.name}")
        return math.prod(self.shape) // block * width


def read_headers(checkpoint: Path) -> tuple[dict, dict]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    mapping, headers = index["weight_map"], {}
    for shard in sorted(set(mapping.values())):
        path = checkpoint / shard
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 64 * 1024 * 1024:
                raise ValueError(f"Oversized header: {shard}")
            header = json.loads(stream.read(length))
        spans = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in headers or mapping.get(name) != shard:
                raise ValueError(f"Duplicate/unindexed tensor: {name}")
            start, end = entry["data_offsets"]
            count = math.prod(entry["shape"]) * SOURCE_WIDTH[entry["dtype"]]
            if end - start != count or not 0 <= start <= end <= path.stat().st_size - 8 - length:
                raise ValueError(f"Invalid payload bounds: {name}")
            spans.append((start, end))
            headers[name] = entry
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError(f"Overlapping tensors: {shard}")
    if set(headers) != set(mapping):
        raise ValueError("Checkpoint index/header names differ")
    return mapping, headers


def choose_storage(name: str, entry: dict) -> int:
    """Protect routers, recurrent parameters and norms; retain BF16 tower."""
    shape = entry["shape"]
    if name.startswith("audio_decoder"):
        return 30 if entry["dtype"] == "BF16" and len(shape) > 1 else 0
    sensitive = name.endswith((".mlp.gate.weight", ".shared_expert_gate.weight"))
    if len(shape) != 2 or sensitive or "norm" in name:
        return 0
    return Q8_0


def plan_tensors(headers: dict, layers: set[int] | None, experts_only: bool) -> list[TensorPlan]:
    banks: dict[tuple[int, str], dict[int, str]] = {}
    plans = []
    for name, entry in headers.items():
        if name.endswith("_scale_inv"):
            continue
        match = EXPERT.fullmatch(name)
        if match:
            layer, expert, projection = int(match[1]), int(match[2]), match[3]
            if layers is not None and layer not in layers:
                continue
            scale = headers.get(name + "_scale_inv")
            expected = [2048, 512] if projection == "down_proj" else [512, 2048]
            if entry["dtype"] != "F8_E4M3" or entry["shape"] != expected:
                raise ValueError(f"Unsupported expert geometry: {name}")
            if not scale or scale["dtype"] != "F32" or scale["shape"] != [n // 128 for n in expected]:
                raise ValueError(f"Missing/invalid scale: {name}")
            slots = banks.setdefault((layer, projection), {})
            if expert in slots or not 0 <= expert < 256:
                raise ValueError(f"Invalid expert slot: {name}")
            slots[expert] = name
        elif not experts_only:
            plans.append(TensorPlan(dest_name(name), tuple(entry["shape"]),
                                    choose_storage(name, entry), (name,)))
    expected_layers = set(range(40)) if layers is None else layers
    if set(banks) != {(layer, p) for layer in expected_layers for p in ("gate_proj", "up_proj", "down_proj")}:
        raise ValueError("Missing expert banks")
    for (layer, projection), slots in sorted(banks.items()):
        if set(slots) != set(range(256)):
            raise ValueError(f"Incomplete bank L{layer}/{projection}")
        shape = (256, *headers[slots[0]]["shape"])
        name = f"model.layers.{layer}.mlp.experts_bank.{projection}.weight"
        plans.append(TensorPlan(name, shape, Q4_K, tuple(slots[e] for e in range(256))))
    if len({p.name for p in plans}) != len(plans):
        raise ValueError("Duplicate destination names")
    return sorted(plans, key=lambda p: p.name)


class SourceReader:
    """Keep one read-only safetensors mapping open; tensors are row-sliced."""

    def __init__(self, checkpoint: Path, mapping: dict):
        self.checkpoint, self.mapping = checkpoint, mapping
        self.shard = None
        self.handle = None

    def get(self, name: str):
        shard = self.mapping[name]
        if self.shard != shard:
            self.close()
            self.handle = safe_open(str(self.checkpoint / shard), framework="pt", device="cpu")
            self.handle.__enter__()
            self.shard = shard
        return self.handle.get_slice(name)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.__exit__(None, None, None)
            self.handle = None
            self.shard = None


def source_chunks(plan: TensorPlan, reader: SourceReader, rows: int) -> Iterator[torch.Tensor]:
    if rows <= 0 or rows % 128:
        raise ValueError("Row chunks must be positive multiples of 128")
    for source in plan.sources:
        tensor = reader.get(source)
        shape = tensor.get_shape()
        if len(shape) == 1:
            value = tensor[:].float().contiguous()
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite source: {source}")
            yield value
            continue
        for start in range(0, shape[0], rows):
            stop = min(start + rows, shape[0])
            value = reader.get(source)[start:stop]
            if plan.storage == Q4_K:
                scale = reader.get(source + "_scale_inv")[start // 128:stop // 128]
                if not torch.isfinite(scale).all():
                    raise ValueError(f"Nonfinite scale: {source}")
                value = fp8_block_reconstruct(value, scale)
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite source: {source}")
            yield value


def payload_chunks(plan: TensorPlan, reader: SourceReader, rows: int = 128) -> Iterator[bytes]:
    chunks = source_chunks(plan, reader, rows)
    if plan.storage in TYPE_SIZES:
        # ctypes releases the GIL. At most eight FP32 chunks are queued;
        # yielding futures in order preserves byte-for-byte bank ordering.
        pending = deque()
        with ThreadPoolExecutor(max_workers=4) as executor:
            for value in chunks:
                pending.append(executor.submit(quantize, value.float().numpy(), plan.storage))
                if len(pending) == 8:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
    else:
        for value in chunks:
            if plan.storage == 30:
                yield value.contiguous().view(torch.uint16).numpy().tobytes()
            else:
                yield value.float().contiguous().numpy().tobytes()


def source_identity(checkpoint: Path, mapping: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "data/source/drama3-dd64a85-sha256.json").read_text(encoding="utf-8"))
    if pins.get("revision") != REVISION:
        raise ValueError("Source pin revision mismatch")
    expected = pins["files"]
    if not set(mapping.values()) <= expected.keys():
        raise ValueError("Unpinned checkpoint shard")
    identities = {}
    for name, digest in sorted(expected.items()):
        if Path(name).name != name or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid source pin entry")
        actual = sha_file(checkpoint / name)
        if actual != digest:
            raise ValueError(f"Source hash mismatch: {name}")
        identities[name] = actual
    return identities


def configure_writer(path: Path, checkpoint: Path, plans: list[TensorPlan], complete: bool):
    writer = StreamingGGUFWriter(path)
    writer.add_kv("fish_drama3.contract", T_STRING, "fish_drama3-v0.1-direct")
    writer.add_kv("fish_drama3.recipe", T_STRING, RECIPE)
    writer.add_kv("fish_drama3.revision", T_STRING, REVISION)
    writer.add_kv("fish_drama3.complete", T_BOOL, complete)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        writer.add_kv(f"fish_drama3.resource.{name}", T_STRING,
                      (checkpoint / name).read_text(encoding="utf-8"))
    for plan in plans:
        writer.add_tensor(plan.name, list(plan.shape), plan.storage, plan.nbytes, Path("."))
    return writer


def verify_output(path: Path, plans: list[TensorPlan]) -> None:
    from gguf import GGUFReader

    result = GGUFReader(str(path))
    found = {tensor.name: tensor for tensor in result.tensors}
    if set(found) != {plan.name for plan in plans}:
        raise ValueError("Output tensor set mismatch")
    for plan in plans:
        tensor = found[plan.name]
        if tuple(tensor.shape) != tuple(reversed(plan.shape)) or int(tensor.tensor_type) != plan.storage:
            raise ValueError(f"Output shape/type mismatch: {plan.name}")
        if tensor.data.nbytes != plan.nbytes:
            raise ValueError(f"Output payload-size mismatch: {plan.name}")


def convert(checkpoint: Path, output: Path, layers: set[int] | None = None,
            experts_only: bool = False, include_codec: bool = True) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    torch.set_num_threads(1)  # tiny row blocks; native quantizer has four bounded workers
    mapping, headers = read_headers(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "fish_omni":
        raise ValueError("Expected fish_omni checkpoint")
    plans = plan_tensors(headers, layers, experts_only)
    needed = sum(p.nbytes for p in plans) + (3 * 2**30 if include_codec else 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < needed + 5 * 2**30:
        raise OSError("Insufficient disk for direct output plus 5 GiB margin")
    source_hashes = source_identity(checkpoint, mapping)
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(exist_ok=False)
    complete = layers is None and not experts_only and include_codec
    writer = configure_writer(staging / "model.gguf", checkpoint, plans, complete)
    reader = SourceReader(checkpoint, mapping)
    started = time.monotonic()
    try:
        writer.begin()
        for index, plan in enumerate(plans):
            writer.stream_payload(plan.name, payload_chunks(plan, reader))
            if index % 10 == 0 or index + 1 == len(plans):
                print(f"Converted {index + 1}/{len(plans)} tensors: {plan.name}", flush=True)
        record = writer.finish()
        reader.close()
        verify_output(staging / "model.gguf", plans)
        codec = write_codec(checkpoint, staging) if include_codec else None
        manifest = make_manifest(staging, record, codec, source_hashes, complete, started)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.copy2(checkpoint / "LICENSE", staging / "LICENSE")
        staging.rename(output)
        return manifest
    finally:
        reader.close()
        writer.abort()


def make_manifest(staging: Path, record: dict, codec: dict | None, sources: dict,
                  complete: bool, started: float) -> dict:
    return {"contract": "fish_drama3-v0.1-direct", "recipe": RECIPE,
            "revision": REVISION, "complete": complete, "status": "validated-conversion",
            "model": record, "codec": codec, "source_files": sources,
            "model_sha256": sha_file(staging / "model.gguf"),
            "quantizer_sha256": sha_file(DLL_PATH),
            "seconds": time.monotonic() - started,
            "quality_qualified": False, "runtime_qualified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New immutable bundle directory")
    parser.add_argument("--layers", help="Inclusive expert layer range, e.g. 4-4")
    parser.add_argument("--experts-only", action="store_true")
    parser.add_argument("--no-codec", action="store_true")
    args = parser.parse_args()
    layers = None
    if args.layers:
        lower, upper = (int(v) for v in args.layers.split("-"))
        if not 0 <= lower <= upper < 40:
            parser.error("--layers must be within 0-39")
        layers = set(range(lower, upper + 1))
    manifest = convert(args.checkpoint, args.output, layers, args.experts_only, not args.no_codec)
    print(f"Published conversion: {args.output}; {manifest['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
