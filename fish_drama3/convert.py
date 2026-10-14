#!/usr/bin/env python3
"""fish_drama3 converter: Stage A FP32 baseline + Stage B bank spools + GGUF.

Implements P2_CONVERTER_SPEC against frozen P1_REVIEW §6/§7.
Stage A: mirror-eight FP32 safetensors + separate codec file + ledger,
  streaming writes, resume by re-hash, atomic bundle publish.
Stage B: per-layer temp expert-bank spools -> single FP32 GGUF via the
  bounded streaming writer (gguf-py's writer retains all arrays: unusable
  at 120 GiB, see gguf_stream.py).
Core rule: expert IDs parse NUMERICALLY; bank slot == expert E at byte
offset E*O*I*itemsize. Duplicates or incomplete banks are hard errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

CONTRACT = "fish_drama3-v0.1"
BACKBONE_STRIP = "model.language_model."
TOWER_RENAMES = (
    ("audio_decoder.codebook_embeddings.", "codebook_embeddings."),
    ("audio_decoder_mtp_shared.layers.", "shared_layers."),
    ("audio_decoder_mtp.layers.", "residual_layers."),
    ("audio_decoder_mtp.heads.", "codebook_heads."),
)
EXP_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(_scale_inv)?$")


def parse_expert_key(name: str) -> tuple[int, int, str, bool] | None:
    """(layer, expert, proj, is_scale) with NUMERIC ids, else None.

    '10' parses as 10, never as lexicographic neighbor of '1'.
    """
    m = EXP_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4) is not None


def fp8_block_reconstruct(w_f8: torch.Tensor, s_f32: torch.Tensor) -> torch.Tensor:
    """Apply source FP8 scales ONCE. Geometry derived per tensor."""
    w = w_f8.to(torch.float32)
    s = s_f32.to(torch.float32)
    ob, ib = w.shape[0] // 128, w.shape[1] // 128
    assert w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0, tuple(w.shape)
    assert list(s.shape) == [ob, ib], (list(s.shape), [ob, ib])
    r = (w.reshape(ob, 128, ib, 128) * s.reshape(ob, 1, ib, 1)).reshape(
        w.shape[0], w.shape[1])
    if not torch.isfinite(r).all():
        raise FloatingPointError("nonfinite reconstruction rejected")
    return r


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):
            h.update(chunk)
    return h.hexdigest()


def dest_name(src: str) -> str:
    if src.startswith(BACKBONE_STRIP):
        return "model." + src[len(BACKBONE_STRIP):]
    for old, new in TOWER_RENAMES:
        if src.startswith(old):
            return new + src[len(old):]
    if src == "lm_head.weight":
        return src
    raise ValueError(f"no contract rename for {src}")


def select_name(name: str, components: set[str],
                layers: tuple[int, int] | None) -> bool:
    """Stage A selection predicate (shared by planner and preflight)."""
    if name.endswith("_scale_inv"):
        return False  # provenance only, never runtime
    parsed = parse_expert_key(name)
    if parsed is not None:
        layer = parsed[0]
        return "experts" in components and (layers is None or layers[0] <= layer <= layers[1])
    if name.startswith("audio_decoder"):
        return "tower" in components
    if name == "lm_head.weight" or "embed_tokens" in name:
        return "embeddings" in components
    if "codec" not in name:
        return "dense" in components
    return False


class ShardPool:
    """One open safe_open handle per source shard for a streaming pass."""

    def __init__(self, ckpt: Path, shards: list[str]):
        self.ckpt = ckpt
        self.files = {s: safe_open(str(ckpt / s), framework="pt", device="cpu")
                      for s in shards}

    def close(self) -> None:
        for f in self.files.values():
            f.__exit__(None, None, None)

    def get(self, shard: str, name: str) -> torch.Tensor:
        return self.files[shard].get_tensor(name)


def load_weight_map(ckpt: Path) -> dict[str, str]:
    return json.loads((ckpt / "model.safetensors.index.json").read_text(
        encoding="utf-8"))["weight_map"]


def read_header(ckpt: Path, shard: str) -> dict:
    with open(ckpt / shard, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen))


def plan_partitions(ckpt: Path, components: set[str],
                    layers: tuple[int, int] | None) -> list[dict]:
    """Mirror-eight Stage A plan from the index (no payloads read)."""
    wmap = load_weight_map(ckpt)
    parts: dict[str, list[str]] = {}
    for name, shard in wmap.items():
        if select_name(name, components, layers):
            parts.setdefault(shard, []).append(name)
    return [{"shard": s, "names": sorted(ns)} for s, ns in sorted(parts.items())]


def write_partition(ckpt: Path, pool: ShardPool, part: dict, out_tmp: Path) -> dict:
    """Stream one partition to a precomputed-header safetensors file."""
    wmap = load_weight_map(ckpt)
    resolved = []
    offset = 0
    for src in part["names"]:
        shard = wmap[src]
        with open(ckpt / shard, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            e = json.loads(f.read(hlen))[src]
        n = 1
        for d in e["shape"]:
            n *= d
        size = n * 4
        resolved.append((dest_name(src), e["shape"], offset, size, src))
        offset += size
    header = {d: {"dtype": "F32", "shape": s, "data_offsets": [o, o + sz]}
              for d, s, o, sz in ((r[0], r[1], r[2], r[3]) for r in resolved)}
    hbytes = json.dumps(header).encode()
    with open(out_tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hbytes)))
        out.write(hbytes)
        for dest, shape, off, size, src in resolved:
            t = pool.get(wmap[src], src)
            if t.dtype == torch.float8_e4m3fn:
                t = fp8_block_reconstruct(t, pool.get(wmap[src + "_scale_inv"], src + "_scale_inv"))
            else:
                t = t.to(torch.float32)
            blob = t.numpy().tobytes()
            assert len(blob) == size, (dest, len(blob), size)
            out.write(blob)
            del t, blob
    return {"file": out_tmp.name, "bytes": out_tmp.stat().st_size,
            "tensors": len(resolved), "sha256": sha_file(out_tmp)}


STYPE = {torch.float32: ("F32", 4), torch.bfloat16: ("BF16", 2),
         torch.float16: ("F16", 2), torch.bool: ("BOOL", 1)}


def write_codec(ckpt: Path, staging: Path) -> dict:
    """Fold 60 pairs + passthrough rest -> separate codec file (streaming).

    Amended contract (b): learned + folded weights FP32; structural buffers
    (bool masks, BF16 RoPE tables) retain source dtype, explicit in ledger.
    """
    d = torch.load(str(ckpt / "codec.pth"), map_location="cpu", weights_only=True)
    pairs = sorted({k.rsplit(".original", 1)[0] for k in d
                    if ".parametrizations.weight.original0" in k})
    assert len(pairs) == 60, f"codec pair count {len(pairs)}"
    consumed = {p + ".original0" for p in pairs} | {p + ".original1" for p in pairs}
    rest = sorted(k for k in d if k not in consumed)
    assert len(pairs) * 2 + len(rest) == len(d), "codec coverage must be total"
    items = []
    for p in pairs:
        g, v = d[p + ".original0"], d[p + ".original1"]
        assert g.shape[0] == v.shape[0], p
        if not (torch.isfinite(g).all() and torch.isfinite(v).all()):
            raise FloatingPointError(f"nonfinite codec source {p}")
        items.append(("fold", p.replace(".parametrizations", ""), tuple(v.shape), (g.float(), v.float())))
    for k in rest:
        t = d[k]
        if t.dtype not in STYPE:
            raise ValueError(f"unexpected codec dtype {t.dtype} for {k}")
        if t.dtype != torch.bool and not torch.isfinite(t).all():
            raise FloatingPointError(f"nonfinite codec source {k}")
        items.append(("pass", k, tuple(t.shape), t))
    names = [it[1] for it in items]
    assert len(set(names)) == len(names), "codec names must be unique"
    header, offset, hashes = {}, 0, {}
    for kind, name, shape, t in items:
        dt = "F32" if kind == "fold" else STYPE[t.dtype][0]
        es = 4 if kind == "fold" else STYPE[t.dtype][1]
        n = 1
        for dd in shape:
            n *= dd
        size = n * es
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    hbytes = json.dumps(header).encode()
    out_tmp = staging / "codec.fp32.safetensors.tmp"
    with open(out_tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hbytes)))
        out.write(hbytes)
        for kind, name, shape, t in items:
            if kind == "fold":
                g, v = t
                norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape(-1, *([1] * (v.dim() - 1)))
                if not torch.isfinite(norm).all() or bool((norm == 0).any()):
                    raise FloatingPointError(f"zero/nonfinite norm {name}")
                blob = (g * v / norm).numpy().tobytes()
            elif t.dtype == torch.bfloat16:
                blob = t.contiguous().view(torch.uint16).numpy().tobytes()  # bit-exact
            else:
                blob = t.numpy().tobytes()
            assert len(blob) == (header[name]["data_offsets"][1] - header[name]["data_offsets"][0])
            out.write(blob)
            hashes[name] = hashlib.sha256(blob).hexdigest()
            del blob
    del items
    del d
    out_final = staging / "codec.fp32.safetensors"
    rec = {"file": out_final.name, "bytes": out_tmp.stat().st_size,
           "tensors": len(header), "sha256": sha_file(out_tmp),
           "tensor_hashes": hashes,
           "dtypes": {dt: sum(1 for v in header.values() if v["dtype"] == dt)
                      for dt in {v["dtype"] for v in header.values()}}}
    out_tmp.rename(out_final)
    with safe_open(str(out_final), framework="pt", device="cpu") as f:
        assert len(set(f.keys())) == len(header), "codec reload count"
    return rec


def load_or_init_ledger(staging: Path, revision: str, components: set[str],
                        layers: tuple[int, int] | None, resume: bool) -> dict:
    staging.mkdir(parents=True, exist_ok=True)
    ledger_path = staging / "ledger.json"
    want = {"contract": CONTRACT, "revision": revision,
            "components": sorted(components), "layers": list(layers) if layers else None}
    if resume and ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        for k, v in want.items():
            if ledger.get(k) != v:
                raise ValueError(f"resume identity mismatch on {k}")
        return ledger
    return dict(want, files=[])


def convert_stage_a(ckpt: Path, staging: Path, revision: str, components: set[str],
                    layers: tuple[int, int] | None, resume: bool) -> dict:
    ledger = load_or_init_ledger(staging, revision, components, layers, resume)
    ledger_path = staging / "ledger.json"
    plan = plan_partitions(ckpt, components, layers)
    for part in plan:
        out_final = staging / f"stageA-{part['shard'].replace('.safetensors', '.fp32.safetensors')}"
        meta = next((f for f in ledger["files"] if f["file"] == out_final.name), None)
        if resume and out_final.exists() and meta and sha_file(out_final) == meta["sha256"]:
            print(f"resume-skip intact {out_final.name}")
            continue
        pool = ShardPool(ckpt, [part["shard"]])
        try:
            tmp = staging / (out_final.name + ".tmp")
            rec = write_partition(ckpt, pool, part, tmp)
            rec["file"] = out_final.name
            assert sha_file(tmp) == rec["sha256"]
            with safe_open(str(tmp), framework="pt", device="cpu") as f:
                assert set(f.keys()) == {dest_name(s) for s in part["names"]}, \
                    "reload key mismatch"
            tmp.rename(out_final)
            ledger["files"] = [f for f in ledger["files"] if f["file"] != out_final.name]
            ledger["files"].append(rec)
            ledger_path.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
        finally:
            pool.close()
    if "codec" in components:
        codec_done = next((f for f in ledger["files"] if f["file"] == "codec.fp32.safetensors"), None)
        codec_path = staging / "codec.fp32.safetensors"
        if not (resume and codec_done and codec_path.exists()
                and sha_file(codec_path) == codec_done["sha256"]):
            rec = write_codec(ckpt, staging)
            ledger["files"] = [f for f in ledger["files"] if f["file"] != rec["file"]]
            ledger["files"].append(rec)
            ledger_path.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
        else:
            print("resume-skip intact codec.fp32.safetensors")
    return ledger


def verify_bundle_complete(staging: Path, ckpt: Path, components: set[str],
                           layers: tuple[int, int] | None) -> bool:
    """Refuse partial bundles: every planned file present + ledger-complete."""
    ledger_path = staging / "ledger.json"
    if not ledger_path.exists():
        return False
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    want = {f"stageA-{p['shard'].replace('.safetensors', '.fp32.safetensors')}"
            for p in plan_partitions(ckpt, components, layers)}
    if "codec" in components:
        want.add("codec.fp32.safetensors")
    have = {f["file"] for f in ledger.get("files", [])}
    if want != have:
        return False
    return all((staging / f).exists() for f in want)


def estimate_bytes(ckpt: Path, components: set[str],
                   layers: tuple[int, int] | None) -> dict:
    """Exact FP32 bytes from headers (no payloads) + codec zip census."""
    wmap = load_weight_map(ckpt)
    shards = sorted({s for n, s in wmap.items() if select_name(n, components, layers)})
    model_fp32, n_tensors = 0, 0
    for shard in shards:
        header = read_header(ckpt, shard)
        for name, e in header.items():
            if name == "__metadata__" or not select_name(name, components, layers):
                continue
            n = 1
            for d in e["shape"]:
                n *= d
            model_fp32 += n * 4
            n_tensors += 1
    import zipfile
    codec_bytes = 0
    if "codec" in components:
        with zipfile.ZipFile(ckpt / "codec.pth") as z:
            codec_bytes = sum(i.file_size for i in z.infolist()
                              if not i.is_dir() and i.filename != "modded_dac_msstftd_step_1380000/data.pkl"
                              and i.filename != "modded_dac_msstftd_step_1380000/byteorder")
    gguf = int((model_fp32 + codec_bytes) * 1.01)
    spool_peak = 3 * 1024**3  # one full layer bank set (gate+up+down F32)
    return {"model_fp32": model_fp32, "tensors": n_tensors, "codec": codec_bytes,
            "gguf": gguf, "spool_peak": spool_peak,
            "stage_total": model_fp32 + codec_bytes,
            "grand_total": model_fp32 + codec_bytes + gguf + spool_peak}


def preflight(dest_dir: Path, need: dict, margin: int = 50 * 1024**3) -> None:
    free = shutil.disk_usage(dest_dir).free
    total = need["grand_total"]
    print(f"preflight: need={total / 2**30:.1f}GiB margin={margin / 2**30:.0f}GiB "
          f"free={free / 2**30:.1f}GiB")
    if free < total + margin:
        raise OSError(f"insufficient disk: need {total} + margin {margin}, free {free}")


def assemble_banks(stage_a_dir: Path, layers: list[int], tmpdir: Path) -> dict[str, Path]:
    """Per-layer temp spool files: E-ordered banks. Duplicates/incomplete -> error."""
    idx: dict[str, str] = {}
    for fp in sorted(stage_a_dir.glob("stageA-*.fp32.safetensors")):
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in idx:
                    raise ValueError(f"duplicate tensor across partitions: {k}")
                idx[k] = str(fp)
    spools: dict[str, Path] = {}
    for layer in layers:
        for proj, (O, I) in (("gate_proj", (512, 2048)), ("up_proj", (512, 2048)),
                             ("down_proj", (2048, 512))):
            experts: dict[int, str] = {}
            for name, fp in idx.items():
                m = re.fullmatch(rf"model\.layers\.{layer}\.mlp\.experts\.(\d+)\.{proj}\.weight", name)
                if m:
                    eid = int(m.group(1))  # NUMERIC, never lexicographic
                    if eid in experts:
                        raise ValueError(f"duplicate bank slot L{layer}/{proj}/{eid}")
                    experts[eid] = fp
            missing = set(range(256)) - set(experts)
            if missing:
                raise ValueError(f"incomplete bank L{layer}/{proj}: missing {sorted(missing)[:8]}")
            spool = tmpdir / f"L{layer}.{proj}.spool.bin"
            with open(spool, "wb") as out:
                for e in range(256):  # numeric order -> E*O*I*4 offsets
                    with safe_open(experts[e], framework="pt", device="cpu") as f:
                        blob = f.get_tensor(
                            f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight").numpy().tobytes()
                    assert len(blob) == O * I * 4, (layer, proj, e)
                    assert out.tell() == e * O * I * 4, "offset drift"
                    out.write(blob)
            spools[f"{layer}.{proj}"] = spool
    return spools


def assemble_gguf(spools: dict[str, Path], extra: dict[str, Path], out_path: Path,
                  meta: dict) -> dict:
    """Single GGUF via the bounded streaming writer (never accumulates payloads)."""
    try:
        from fish_drama3.gguf_stream import StreamingGGUFWriter
    except ImportError:
        from gguf_stream import StreamingGGUFWriter
    w = StreamingGGUFWriter(out_path)
    w.add_kv("fish_drama3.contract", 8, CONTRACT)
    for k, v in meta.items():
        w.add_kv(f"fish_drama3.{k}", 8, str(v))
    for key in sorted(spools):
        layer, proj = key.split(".")
        O, I = (512, 2048) if proj in ("gate_proj", "up_proj") else (2048, 512)
        nbytes = 256 * O * I * 4
        assert spools[key].stat().st_size == nbytes, key
        name = f"model.layers.{layer}.mlp.experts_bank.{proj}.weight"
        w.add_tensor(name, [256, O, I], 0, nbytes, spools[key], 0)
    for name, fp in sorted(extra.items()):
        with open(fp, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        for k in sorted(header):
            if k == "__metadata__":
                continue
            if re.fullmatch(r"model\.layers\.\d+\.mlp\.experts\.\d+\."
                            r"(gate_proj|up_proj|down_proj)\.weight", k):
                continue  # replaced by E-ordered banks above
            e = header[k]
            assert e["dtype"] == "F32", (k, e["dtype"])
            nbytes = e["data_offsets"][1] - e["data_offsets"][0]
            w.add_tensor(k, e["shape"], 0, nbytes, fp, 8 + hlen + e["data_offsets"][0])
    rec = w.finalize()
    rec["sha256"] = sha_file(out_path)
    return {"file": out_path.name, "bytes": rec["bytes"], "tensors": rec["tensors"],
            "sha256": rec["sha256"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="fish_drama3 Stage A/B converter")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-bundle", required=True)
    ap.add_argument("--revision", required=True, help="pinned source revision (no default)")
    ap.add_argument("--components", nargs="+",
                    default=["experts", "dense", "tower", "codec", "embeddings"])
    ap.add_argument("--layers", default=None, help="Lo-Hi expert layer range")
    ap.add_argument("--stage", default="both", choices=["A", "B", "both"])
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    layers = tuple(int(x) for x in args.layers.split("-")) if args.layers else None
    if layers and not (0 <= layers[0] <= layers[1] <= 39):
        raise ValueError("--layers Lo-Hi within 0-39")
    bundle = Path(args.output_bundle)
    staging = bundle.parent / (bundle.name + ".staging")
    need = estimate_bytes(ckpt, set(args.components), layers)
    print(f"estimate: {json.dumps({k: v for k, v in need.items()})}")
    preflight(bundle.parent, need)
    try:
        ledger = convert_stage_a(ckpt, staging, args.revision, set(args.components),
                                 layers, resume=not args.no_resume)
        (staging / "ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
        if bundle.exists():
            raise FileExistsError(f"refusing to overwrite qualified bundle {bundle}")
        assert verify_bundle_complete(staging, ckpt, set(args.components), layers), \
            "partial bundle refused"
        staging.rename(bundle)
        print(f"stage A published: {bundle} ({len(ledger['files'])} files)")
    except Exception:
        print(f"FAILED - staging left at {staging} (never half-published)")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
