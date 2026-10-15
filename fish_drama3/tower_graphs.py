"""Optional audio-tower CUDA graphs for the hybrid CPU/GPU backbone.

Only the CUDA-resident audio decoder is captured. The CPU expert bridge and
backbone remain eager. Capture preserves the caller's RNG and prefilled cache
rows; finite checks cover every sampling head and synchronize once per frame.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterator

import torch

from fish_speech.models.text2semantic.fish_omni.tower import (
    RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, RAS_WINDOW_FRAMES,
    sample_top_k_top_p,
)

KV_BUCKETS = (256, 512, 1024, 2048, 4096, 8192, 10240)


def tower_attentions(tower: Any) -> Iterator[Any]:
    for layer in (*tower.shared_layers, *tower.residual_layers):
        yield layer.transformer_layer.attention


def cache_signature(tower: Any) -> tuple[int, ...]:
    """Changing cache allocations invalidates every captured pointer."""
    return tuple(value for attention in tower_attentions(tower)
                 for cache in (attention.k_cache, attention.v_cache)
                 for value in (cache.data_ptr(), cache.shape[2]))


def checked_decode(tower: Any, logits: torch.Tensor, hidden: torch.Tensor,
                   positions: torch.Tensor, frames: torch.Tensor,
                   history: torch.Tensor, temperature: Any, top_k: Any,
                   top_p: Any, finite: torch.Tensor) -> torch.Tensor:
    """Pinned upstream decode with its per-head checks aggregated on device."""
    finite.fill_(True)

    def sample(values: torch.Tensor, temp: Any, k: Any, p: Any) -> torch.Tensor:
        finite.logical_and_(torch.isfinite(values).all())
        return sample_top_k_top_p(values, temp, k, p, check_finite=False)

    candidates = tower.main_candidates
    candidate_logits = logits[:, candidates]
    first = candidates[sample(candidate_logits, temperature, top_k, top_p)]
    retry = candidates[sample_top_k_top_p(candidate_logits, RAS_TEMPERATURE,
                       RAS_TOP_K, RAS_TOP_P, check_finite=False)]
    repeated = (history[-RAS_WINDOW_FRAMES:] == first[:, None]).any(-1)
    repeated &= tower.is_semantic(first)
    main = torch.where(repeated, retry, first)
    first_code = (main - tower.semantic_begin).clamp(0, tower.config.vocab_size - 1)
    semantic = tower.is_semantic(main)
    hidden = tower.run_shared(hidden, first_code, semantic, positions, frames)
    codes = [first_code]
    for layer, head in zip(tower.residual_layers, tower.codebook_heads):
        hidden = layer(hidden, codes[-1], semantic, positions)
        codes.append(sample(head(hidden), temperature, top_k, top_p))
    return torch.stack((main, *codes), dim=-1)


class TowerDecodeGraphs:
    """Lazy, single-row tower graphs; no model/backbone graph is required."""

    def __init__(self, tower: Any, original: Callable[..., torch.Tensor]) -> None:
        self.tower, self.original = tower, original
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.signature: tuple[int, ...] = ()
        self.pool: Any = None
        self.capture_stream: Any = None
        self.capture_seconds = 0.0
        self.replays = 0

    def initialize(self, logits: torch.Tensor, hidden: torch.Tensor,
                   frames: torch.Tensor) -> None:
        self.logits, self.hidden = torch.zeros_like(logits), torch.zeros_like(hidden)
        self.frame = torch.zeros_like(frames)
        self.position = torch.zeros(1, dtype=torch.long, device=self.tower.device)
        self.history = torch.full((RAS_WINDOW_FRAMES,), -1, dtype=torch.long,
                                  device=self.tower.device)
        self.temperature = torch.ones((), device=self.tower.device)
        self.top_k = torch.full((), 30, dtype=torch.long, device=self.tower.device)
        self.top_p = torch.ones((), device=self.tower.device)
        self.finite = torch.ones((), dtype=torch.bool, device=self.tower.device)
        self.sampled = torch.zeros_like(frames)
        eos = torch.tensor([self.tower.eos_id], device=self.tower.device)
        band = torch.arange(self.tower.semantic_begin, self.tower.semantic_end + 1,
                            device=self.tower.device)
        self.tower.main_candidates = torch.cat((eos, band)).sort().values
        self.signature = cache_signature(self.tower)
        self.graphs.clear()
        self.pool = torch.cuda.graph_pool_handle()
        # cuBLAS retains a workspace per stream. Reuse this stream for every
        # bucket's warmup and capture instead of paying that cost seven times.
        self.capture_stream = torch.cuda.Stream(device=self.tower.device)

    def run(self) -> None:
        sampled = checked_decode(self.tower, self.logits, self.hidden,
                                 self.position, self.frame, self.history,
                                 self.temperature, self.top_k, self.top_p, self.finite)
        self.sampled.copy_(sampled)

    def capture(self, bucket: int) -> None:
        """Warm up at row zero, restoring that row and both RNG streams."""
        started = time.perf_counter()
        attentions = list(tower_attentions(self.tower))
        saved = [(cache, cache[:, :, :1].clone()) for attention in attentions
                 for cache in (attention.k_cache, attention.v_cache)]
        device = self.tower.device
        self.position.zero_()
        try:
            for attention in attentions:
                attention.static_kv_len = bucket
            with torch.random.fork_rng(devices=[device]):
                stream = self.capture_stream
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    self.run()
                    self.run()
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                    self.run()
                self.graphs[bucket] = graph
        finally:
            for cache, row in saved:
                cache[:, :, :1].copy_(row)
            for attention in attentions:
                attention.static_kv_len = None
        self.capture_seconds += time.perf_counter() - started

    @torch.inference_mode()
    def __call__(self, logits: torch.Tensor, hidden: torch.Tensor,
                 positions: torch.Tensor, frames: torch.Tensor,
                 history: torch.Tensor, *, temperature: Any, top_k: Any,
                 top_p: Any, check_finite: bool = True) -> torch.Tensor:
        if logits.device.type != "cuda" or len(positions) != 1:
            return self.original(logits, hidden, positions, frames, history,
                                 temperature=temperature, top_k=top_k,
                                 top_p=top_p, check_finite=check_finite)
        signature = cache_signature(self.tower)
        if signature != self.signature:
            self.initialize(logits, hidden, frames)
        capacity = next(tower_attentions(self.tower)).k_cache.shape[2]
        position = int(positions[-1])
        if not 0 <= position < capacity:
            raise ValueError("Tower position exceeds allocated cache capacity")
        bucket = min(next((size for size in KV_BUCKETS if size > position), capacity),
                     capacity)
        if bucket not in self.graphs:
            self.capture(bucket)
        self.logits.copy_(logits)
        self.hidden.copy_(hidden)
        self.position.copy_(positions)
        self.frame.copy_(frames)
        self.history.fill_(-1)
        window = history[-RAS_WINDOW_FRAMES:]
        self.history[RAS_WINDOW_FRAMES - len(window):].copy_(window)
        self.temperature.fill_(temperature)
        self.top_k.fill_(top_k)
        self.top_p.fill_(top_p)
        self.graphs[bucket].replay()
        self.replays += 1
        if check_finite and not self.finite.item():
            raise FloatingPointError(
                "non-finite logits; refusing to sample invalid audio")
        return self.sampled.clone()


def install_tower_graphs(model: Any) -> TowerDecodeGraphs:
    """Explicit opt-in; callers must serialize generation for a model instance."""
    tower = model.audio_decoder
    existing = getattr(tower, "_hybrid_decode_graphs", None)
    if existing is not None:
        return existing
    graphs = TowerDecodeGraphs(tower, tower.decode_step)
    tower.decode_step = graphs
    tower._hybrid_decode_graphs = graphs
    return graphs


def uninstall_tower_graphs(model: Any) -> None:
    """Restore eager decode and release captured graphs after a failed setup."""
    tower = model.audio_decoder
    graphs = getattr(tower, "_hybrid_decode_graphs", None)
    if graphs is not None:
        class_method = getattr(type(tower), "decode_step", None)
        if getattr(graphs.original, "__func__", None) is class_method:
            del tower.decode_step
        else:
            tower.decode_step = graphs.original
        graphs.graphs.clear()
        del tower._hybrid_decode_graphs


@torch.inference_mode()
def prepare_tower_graphs(model: Any) -> dict[str, Any]:
    """Capture all allocated-cache buckets before GPU expert placement.

    Expert placement must measure free VRAM *after* this call, so graph buffers
    are included in its memory budget rather than borrowed from user headroom.
    """
    try:
        return _prepare_tower_graphs(model)
    except BaseException:
        uninstall_tower_graphs(model)
        raise


def _prepare_tower_graphs(model: Any) -> dict[str, Any]:
    tower = model.audio_decoder
    if tower.device.type != "cuda":
        return dict(enabled=False, reason="audio tower is not on CUDA")
    graphs = install_tower_graphs(model)
    attentions = list(tower_attentions(tower))
    if not attentions or attentions[0].k_cache is None:
        raise ValueError("Allocate audio tower caches before preparing graphs")
    torch.cuda.synchronize(tower.device)
    free_before = torch.cuda.mem_get_info(tower.device)[0]
    weight = model.backbone.lm_head.weight
    logits = weight.new_zeros((1, weight.shape[0]))
    hidden = weight.new_zeros((1, weight.shape[1]))
    frames = torch.zeros((1, 1 + tower.config.num_codebooks), dtype=torch.long,
                          device=tower.device)
    graphs.initialize(logits, hidden, frames)
    capacity = attentions[0].k_cache.shape[2]
    for bucket in sorted({min(size, capacity) for size in KV_BUCKETS}):
        graphs.capture(bucket)
    torch.cuda.synchronize(tower.device)
    free_after = torch.cuda.mem_get_info(tower.device)[0]
    return dict(enabled=True, buckets=sorted(graphs.graphs),
                capture_seconds=graphs.capture_seconds,
                gpu_free_bytes_before=free_before, gpu_free_bytes_after=free_after,
                gpu_reserved_bytes=max(0, free_before - free_after))
