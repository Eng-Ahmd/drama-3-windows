"""Hybrid Windows inference with native GGML experts and the pinned Fish model.

Generation returns acoustic codes for waveform decoding through the native codec.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType, dequantize
from tokenizers import Tokenizer
from fish_drama3.gguf_stream import read_resource_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/fish-speech"))

from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt, ensure_drama3_caches, generate_drama3,
)
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder
from fish_speech.models.text2semantic.sampling import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


class ExpertBridge:
    def __init__(self, model: Path, threads: int = 8, gpu_layers: int = 0,
                 headroom_bytes: int | None = None):
        if not 0 <= gpu_layers <= 40 or not 1 <= threads <= 256:
            raise ValueError("Invalid GPU expert layer count or CPU threads")
        if headroom_bytes is not None and not 0 <= headroom_bytes < 2**64:
            raise ValueError("Invalid native GPU headroom byte count")
        build = "drama3-cuda" if gpu_layers else "drama3-inspect"
        dll = ROOT / f"builds/{build}/Release/drama3-experts-bridge.dll"
        self.library = ctypes.CDLL(str(dll))
        self.library.drama3_open.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
        self.library.drama3_open.restype = ctypes.c_void_p
        self.library.drama3_close.argtypes = [ctypes.c_void_p]
        self.library.drama3_close.restype = None
        self.library.drama3_experts.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
        ]
        self.library.drama3_experts.restype = ctypes.c_int
        error = ctypes.create_string_buffer(1024)
        if gpu_layers:
            self.handle = self.open_cuda(model, gpu_layers, headroom_bytes, error)
        else:
            self.handle = self.library.drama3_open(str(model.resolve()).encode(), error, len(error))
        self.threads = threads
        if not self.handle:
            raise RuntimeError(error.value.decode(errors="replace"))

    def open_cuda(self, model: Path, layers: int, headroom: int | None,
                  error: ctypes.Array) -> int:
        arguments = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        values = [str(model.resolve()).encode(), 0, layers]
        name = "drama3_open_cuda"
        if headroom is not None:
            name += "_budget"
            arguments.append(ctypes.c_uint64)
            values.append(headroom)
        try:
            function = getattr(self.library, name)
        except AttributeError as failure:
            raise ValueError("Auto placement requires updated Windows native libraries") from failure
        function.argtypes = arguments + [ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_void_p
        return function(*values, error, len(error))

    def close(self) -> None:
        if self.handle:
            self.library.drama3_close(self.handle)
            self.handle = None

    def forward(self, layer: int, x: torch.Tensor, ids: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        if not self.handle:
            raise RuntimeError("Expert bridge is closed")
        if x.ndim != 2 or x.shape[1] != 2048 or not 1 <= x.shape[0] <= 128:
            raise ValueError("Expert input must have shape [1..128, 2048]")
        if ids.shape != (x.shape[0], 8) or weights.shape != ids.shape:
            raise ValueError("Expert IDs and weights must have shape [tokens, 8]")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expert IDs must be integer tensors")
        values = np.ascontiguousarray(x.float().cpu().numpy())
        indexes = np.ascontiguousarray(ids.int().cpu().numpy())
        mixing = np.ascontiguousarray(weights.float().cpu().numpy())
        output = np.empty_like(values)
        error = ctypes.create_string_buffer(1024)
        status = self.library.drama3_experts(
            self.handle, layer, len(values), values.ctypes.data, indexes.ctypes.data,
            mixing.ctypes.data, output.ctypes.data, self.threads, error, len(error),
        )
        if status:
            raise RuntimeError(error.value.decode(errors="replace"))
        return torch.from_numpy(output).to(device=x.device, dtype=x.dtype)


class NativeExperts(torch.nn.Module):
    def __init__(self, bridge: ExpertBridge, layer: int):
        super().__init__()
        self.bridge, self.layer = bridge, layer

    def forward(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.bridge.forward(self.layer, x, ids, weights)


def copy_parameter(parameter: torch.Tensor, tensor, rows: int = 128) -> None:
    """Dequantize small row blocks, keeping the artifact mapped on CPU."""
    shape = tuple(parameter.shape)
    if tuple(reversed(tensor.shape)) != shape:
        raise ValueError(f"Parameter shape mismatch: {tensor.name}")
    data = tensor.data
    if len(shape) == 1:
        array = np.array(dequantize(data, tensor.tensor_type).reshape(shape), copy=True)
        parameter.copy_(torch.from_numpy(array).to(parameter.device, parameter.dtype))
        return
    for start in range(0, shape[0], rows):
        stop = min(start + rows, shape[0])
        array = np.array(dequantize(data[start:stop], tensor.tensor_type), copy=True)
        value = torch.from_numpy(array)
        parameter[start:stop].copy_(value.to(parameter.device, parameter.dtype))


@torch.inference_mode()
def load_model(bundle: Path, device: str = "cuda:0", threads: int = 8, gpu_layers: int = 0):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("recipe") != "Q4_K-experts_Q8_0-backbone":
        raise ValueError("A complete direct Q4_K/Q8_0 bundle is required")
    reader = GGUFReader(str(bundle / "model.gguf"))
    config = FishOmniConfig.from_dict(json.loads(read_resource_text(reader.fields,
        "fish_drama3.resource.config.json")))
    tokenizer = Tokenizer.from_str(read_resource_text(reader.fields, "fish_drama3.resource.tokenizer.json"))
    if gpu_layers and device != "cuda:0":
        raise ValueError("CUDA expert placement supports cuda:0 only")
    if gpu_layers:
        check_gpu_budget(reader, gpu_layers)
    bridge = ExpertBridge(bundle / "model.gguf", threads, gpu_layers)
    try:
        with torch.device("meta"):
            backbone = FishOmniBackbone(config.text, fp8_experts=False, dtype=torch.bfloat16)
            tower = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
        for index, layer in enumerate(backbone.model.layers):
            layer.mlp.experts = NativeExperts(bridge, index)
        backbone.materialize([torch.device(device)])
        tower.to_empty(device=torch.device(device))
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        copy_parameters(backbone, tower, tensors)
        model = FishOmniModel(config, backbone, tower, tokenizer).eval()
        return model, bridge
    except BaseException:
        bridge.close()
        raise


def check_gpu_budget(reader: GGUFReader, gpu_layers: int) -> None:
    """Reserve backbone/workspace memory before native banks consume device RAM."""
    required = sum(tensor.data.nbytes for tensor in reader.tensors
                   if ".experts_bank." in tensor.name
                   and int(tensor.name.split(".")[2]) >= 40 - gpu_layers)
    available, _ = torch.cuda.mem_get_info(0)
    if required + 9 * 2**30 > available:
        raise ValueError("GPU expert banks plus 8 GiB backbone/workspace reserve "
                         "and 1 GiB headroom exceed available VRAM; reduce GPU layers")


def expert_layer_bytes(bundle: Path) -> list[int]:
    """Read actual quantized expert-bank payload sizes, without dequantization."""
    reader = GGUFReader(str(bundle / "model.gguf"))
    sizes = [0] * 40
    for tensor in reader.tensors:
        if ".experts_bank." in tensor.name:
            layer = int(tensor.name.split(".")[2])
            if not 0 <= layer < len(sizes):
                raise ValueError("Unexpected expert bank layer index")
            sizes[layer] += tensor.data.nbytes
    return sizes


def plan_gpu_layers(sizes: list[int], free: int, total: int,
                    headroom_percent: float = 3) -> dict:
    """Use remaining CUDA memory after dense weights and caches are allocated."""
    if not math.isfinite(headroom_percent) or not 0 < headroom_percent < 100:
        raise ValueError("GPU headroom percentage must be finite and between 0 and 100")
    headroom = math.ceil(total * headroom_percent / 100)
    workspace = 256 * 2**20
    budget = free - headroom - workspace
    if budget < 0:
        raise RuntimeError("Dense model and caches leave insufficient GPU headroom/workspace")
    count = max(count for count in range(len(sizes) + 1) if sum(sizes[len(sizes)-count:]) <= budget)
    return dict(gpu_expert_layers=count, gpu_free_bytes_before_experts=free,
                gpu_total_bytes=total, headroom_percent=headroom_percent,
                headroom_bytes=headroom, workspace_reserve_bytes=workspace,
                expert_budget_bytes=budget, expert_bytes=sum(sizes[len(sizes)-count:]))


def replace_expert_bridge(model: Any, old: ExpertBridge, new: ExpertBridge) -> None:
    layers = model.backbone.model.layers
    previous = [layer.mlp.experts for layer in layers]
    try:
        for index, layer in enumerate(layers):
            layer.mlp.experts = NativeExperts(new, index)
    except BaseException:
        for layer, experts in zip(layers, previous):
            layer.mlp.experts = experts
        new.close()
        raise
    old.close()


def allocate_auto_experts(bundle: Path, model: Any, bridge: ExpertBridge,
                          threads: int, placement: dict, sizes: list[int],
                          automatic: bool = True) -> ExpertBridge:
    planned = placement["gpu_expert_layers"]
    counts = range(planned, 0, -1) if automatic else ([planned] if planned else [])
    for count in counts:
        try:
            replacement = ExpertBridge(bundle / "model.gguf", threads, count,
                                       headroom_bytes=placement["headroom_bytes"])
        except RuntimeError as error:
            if not automatic:
                raise
            placement["last_allocation_error"] = str(error)
            torch.cuda.empty_cache()
            continue
        replace_expert_bridge(model, bridge, replacement)
        placement.update(gpu_expert_layers=count, expert_bytes=sum(sizes[len(sizes)-count:]))
        return replacement
    placement.update(gpu_expert_layers=0, expert_bytes=0)
    return bridge


@torch.inference_mode()
def load_model_auto(bundle: Path, threads: int = 8, headroom_percent: float = 3,
                    cache_capacity: int = 10240) -> tuple[Any, ExpertBridge, dict]:
    return load_model_placed(bundle, threads, None, headroom_percent, cache_capacity)


@torch.inference_mode()
def load_model_placed(bundle: Path, threads: int = 8, gpu_layers: int | None = None,
                      headroom_percent: float = 3,
                      cache_capacity: int = 10240) -> tuple[Any, ExpertBridge, dict]:
    """Stage dense weights/caches first, then fill available VRAM with experts."""
    if isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int) or not 1 <= cache_capacity <= 10240:
        raise ValueError("Auto placement cache capacity must be 1..10240")
    if not math.isfinite(headroom_percent) or not 0 < headroom_percent < 100:
        raise ValueError("GPU headroom percentage must be finite and between 0 and 100")
    if gpu_layers is not None and (isinstance(gpu_layers, bool)
                                  or not isinstance(gpu_layers, int) or not 0 <= gpu_layers <= 40):
        raise ValueError("GPU expert layers must be 0..40")
    model, bridge = load_model(bundle, threads=threads, gpu_layers=0)
    try:
        ensure_drama3_caches(model, cache_capacity)
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        sizes = expert_layer_bytes(bundle)
        placement = plan_gpu_layers(sizes, free, total, headroom_percent)
        if gpu_layers is not None:
            required = sum(sizes[len(sizes)-gpu_layers:])
            if required > placement["expert_budget_bytes"]:
                raise ValueError("Requested GPU expert layers exceed available VRAM after headroom/workspace")
            placement.update(gpu_expert_layers=gpu_layers, expert_bytes=required)
        placement["placement_mode"] = "auto" if gpu_layers is None else "manual"
        placement.update(cache_capacity=cache_capacity, planned_gpu_expert_layers=placement["gpu_expert_layers"])
        bridge = allocate_auto_experts(bundle, model, bridge, threads, placement, sizes,
                                       automatic=gpu_layers is None)
        torch.cuda.synchronize(0)
        placement["gpu_free_bytes_after_experts"] = torch.cuda.mem_get_info(0)[0]
        if placement["gpu_free_bytes_after_experts"] < placement["headroom_bytes"]:
            raise RuntimeError("GPU memory changed during auto placement; headroom is unavailable")
        return model, bridge, placement
    except BaseException:
        bridge.close()
        raise


def copy_parameters(backbone, tower, tensors: dict) -> None:
    targets = list(backbone.named_parameters()) + list(tower.named_parameters())
    expected = {name for name, _ in targets}
    actual = {name for name in tensors if ".experts_bank." not in name}
    if expected != actual:
        raise ValueError(f"Parameter set mismatch: missing={sorted(expected-actual)[:5]} "
                         f"extra={sorted(actual-expected)[:5]}")
    for index, (name, parameter) in enumerate(targets):
        copy_parameter(parameter, tensors[name])
        if index % 100 == 0:
            print(f"Loaded dense/tower {index+1}/{len(targets)}: {name}", flush=True)


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--codes-output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-expert-layers", type=int, default=0)
    parser.add_argument("--reference-codes", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 2048:
        parser.error("--max-frames must be 1..2048")
    if not 0 <= args.gpu_expert_layers <= 40:
        parser.error("--gpu-expert-layers must be 0..40")
    if bool(args.reference_codes) != bool(args.reference_text):
        parser.error("--reference-codes and --reference-text must be provided together")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not 0 < args.top_p <= 1 or args.top_k < 1:
        parser.error("--top-p must be (0,1] and --top-k positive")
    return args


def reference_prompt(args: argparse.Namespace, model) -> torch.Tensor:
    if args.reference_codes is None:
        return build_drama3_prompt(args.text, model.tokenizer, model.config)
    codes = np.load(args.reference_codes, allow_pickle=False)
    if (codes.ndim != 2 or codes.shape[0] != 10 or codes.shape[1] < 1
            or not np.issubdtype(codes.dtype, np.integer)):
        raise ValueError("Reference codes must be integer [10, frames]")
    if np.any(codes < 0) or np.any(codes[0] >= 4096) or np.any(codes[1:] >= 1024):
        raise ValueError("Reference codes exceed semantic/residual vocabulary")
    return build_drama3_prompt(args.text, model.tokenizer, model.config,
                               [args.reference_text], [torch.from_numpy(codes.astype(np.int64))])


def main() -> None:
    args = parse_options()
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"  # CPU expert calls cannot be captured
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, bridge = load_model(args.bundle, threads=args.threads,
                               gpu_layers=args.gpu_expert_layers)
    try:
        prompt = reference_prompt(args, model)
        started = time.perf_counter()
        frames, reason = generate_drama3(model, prompt, max_new_tokens=args.max_frames,
                                         temperature=args.temperature, top_p=args.top_p,
                                         top_k=args.top_k)
        np.save(args.codes_output, frames.numpy(), allow_pickle=False)
        metrics = {"frames": len(frames), "finish_reason": reason,
                   "seconds": time.perf_counter()-started,
                   "peak_torch_vram_bytes": torch.cuda.max_memory_allocated(),
                   "complete": reason == "eos", "seed": args.seed,
                   "text": args.text, "recipe": "Q4_K-experts_Q8_0-backbone",
                   "runtime": "hybrid-gpu-reference_ggml-experts",
                   "gpu_expert_layers": args.gpu_expert_layers,
                   "reference_codes": str(args.reference_codes) if args.reference_codes else None,
                   "reference_text": args.reference_text,
                   "temperature": args.temperature, "top_p": args.top_p,
                   "top_k": args.top_k,
                   "cuda_free_bytes_at_end": torch.cuda.mem_get_info()[0]}
        args.codes_output.with_suffix(".json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics), flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
