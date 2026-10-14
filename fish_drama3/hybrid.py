"""Windows prototype: GGML Q4 CPU experts with the pinned Fish GPU reference.

This is an inference development path, not the standalone C++ release.
Generation returns acoustic codes; waveform decoding is handled separately.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType, dequantize
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/fish-speech"))

from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt, generate_drama3
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder
from fish_speech.models.text2semantic.sampling import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


class ExpertBridge:
    def __init__(self, model: Path, threads: int = 8, gpu_layers: int = 0):
        if not 0 <= gpu_layers <= 40 or not 1 <= threads <= 256:
            raise ValueError("Invalid GPU expert layer count or CPU threads")
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
            self.library.drama3_open_cuda.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            self.library.drama3_open_cuda.restype = ctypes.c_void_p
            self.handle = self.library.drama3_open_cuda(
                str(model.resolve()).encode(), 0, gpu_layers, error, len(error))
        else:
            self.handle = self.library.drama3_open(str(model.resolve()).encode(), error, len(error))
        self.threads = threads
        if not self.handle:
            raise RuntimeError(error.value.decode(errors="replace"))

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
    config = FishOmniConfig.from_dict(json.loads(reader.fields[
        "fish_drama3.resource.config.json"].contents()))
    tokenizer = Tokenizer.from_str(reader.fields["fish_drama3.resource.tokenizer.json"].contents())
    if gpu_layers and device != "cuda:0":
        raise ValueError("CUDA expert prototype currently supports cuda:0 only")
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
