"""Sampling, cache and RNG invariants for the optional hybrid tower graph."""
from pathlib import Path
import os
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream/fish-speech"))

from fish_drama3.tower_graphs import (
    TowerDecodeGraphs, checked_decode, install_tower_graphs, prepare_tower_graphs,
    tower_attentions,
)
import fish_drama3.tower_graphs as tower_graphs
from fish_drama3.watchdog import guarded_job_lock
from fish_speech.models.text2semantic.fish_omni.config import AudioDecoderConfig
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder


def tiny_tower(device: str = "cpu") -> FishOmniAudioDecoder:
    config = AudioDecoderConfig(dim=16, intermediate_size=24, n_head=2,
        n_local_heads=1, head_dim=8, norm_eps=1e-6, rope_base=10000,
        rotary_percent=1.0, vocab_size=8, num_codebooks=3,
        shared_enrichment_layers=1, vq_backbone_codebooks=1)
    tower = FishOmniAudioDecoder(SimpleNamespace(audio_decoder=config,
        semantic_start_token_id=8, semantic_end_token_id=15, eos_token_id=2),
        dtype=torch.float32, device=device).eval()
    tower.setup_caches(512)
    tower.main_candidates = torch.tensor([2, *range(8, 16)], device=device)
    return tower


def inputs(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    return (torch.randn(1, 20, device=device), torch.randn(1, 16, device=device),
            torch.tensor([4], device=device), torch.zeros((1, 4), dtype=torch.long,
            device=device), torch.tensor([8, 9, 8], device=device))


@torch.inference_mode()
@pytest.mark.parametrize("temperature,top_k,top_p", [(0., 1, .1), (.7, 4, .9),
                                                    (1., 30, 1.)])
def test_aggregated_sampling_matches_original_and_rng(
        temperature: float, top_k: int, top_p: float) -> None:
    torch.manual_seed(41)
    tower = tiny_tower()
    values = inputs()
    finite = torch.ones((), dtype=torch.bool)
    torch.manual_seed(123)
    expected = tower.decode_step(*values, temperature=temperature,
                                top_k=top_k, top_p=top_p)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    actual = checked_decode(tower, *values, temperature, top_k, top_p, finite)
    assert finite.item()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@torch.inference_mode()
@pytest.mark.parametrize("bad_head", [-1, 0, 1])
def test_every_sampling_head_contributes_finite_status(bad_head: int) -> None:
    tower, values = tiny_tower(), list(inputs())
    if bad_head == -1:
        values[0][0, 8] = float("nan")
    else:
        tower.codebook_heads[bad_head].weight[0, 0] = float("nan")
    finite = torch.ones((), dtype=torch.bool)
    checked_decode(tower, *values, .7, 4, .9, finite)
    assert not finite.item()


def test_prepare_cpu_tower_does_not_install_graphs() -> None:
    tower = tiny_tower()
    result = prepare_tower_graphs(SimpleNamespace(audio_decoder=tower))
    assert not result["enabled"]
    assert not hasattr(tower, "_hybrid_decode_graphs")


def test_failed_preparation_restores_eager_decoder(monkeypatch) -> None:
    tower = tiny_tower()
    model = SimpleNamespace(audio_decoder=tower)
    original = tower.decode_step
    install_tower_graphs(model)

    def fail(model: object) -> dict:
        raise RuntimeError("capture unavailable")

    monkeypatch.setattr(tower_graphs, "_prepare_tower_graphs", fail)
    with pytest.raises(RuntimeError, match="capture unavailable"):
        prepare_tower_graphs(model)
    assert tower.decode_step == original
    assert "decode_step" not in tower.__dict__
    assert not hasattr(tower, "_hybrid_decode_graphs")


@pytest.mark.skipif(os.environ.get("FISH_DRAMA3_TEST_CUDA_TOWER") != "1"
                    or not torch.cuda.is_available(),
                    reason="opt-in isolated CUDA tower test")
@torch.inference_mode()
def test_cuda_graph_cache_rng_and_sampling_parity() -> None:
    with guarded_job_lock():
        torch.manual_seed(42)
        tower = tiny_tower("cuda:0")
        values = inputs("cuda:0")
        graphs = TowerDecodeGraphs(tower, tower.decode_step)
        graphs.initialize(values[0], values[1], values[3])
        for attention in tower_attentions(tower):
            attention.k_cache.normal_()
            attention.v_cache.normal_()
        saved = [(cache, cache.clone()) for attention in tower_attentions(tower)
                 for cache in (attention.k_cache, attention.v_cache)]
        before_rng = torch.cuda.get_rng_state().clone()
        graphs.capture(256)
        assert torch.equal(torch.cuda.get_rng_state(), before_rng)
        for cache, expected in saved:
            assert torch.equal(cache, expected)
        for seed, temperature, top_k, top_p in (
                (11, 0., 1, .1), (123, .7, 4, .9), (901, 1., 30, 1.)):
            torch.manual_seed(seed)
            expected = tower.decode_step(*values, temperature=temperature,
                                        top_k=top_k, top_p=top_p)
            expected_rng = torch.cuda.get_rng_state().clone()
            torch.manual_seed(seed)
            actual = graphs(*values, temperature=temperature, top_k=top_k, top_p=top_p)
            assert torch.equal(expected, actual)
            assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
        # A later bucket capture must also avoid changing already-prefilled rows.
        graphs.position.fill_(300)
        saved = [(cache, cache.clone()) for attention in tower_attentions(tower)
                 for cache in (attention.k_cache, attention.v_cache)]
        graphs.capture(512)
        for cache, expected in saved:
            assert torch.equal(cache, expected)
        previous_signature = graphs.signature
        tower.setup_caches(1024)
        graphs(*values, temperature=.7, top_k=4, top_p=.9)
        assert graphs.signature != previous_signature
        assert list(graphs.graphs) == [256]
        tower.codebook_heads[-1].weight[0, 0] = float("nan")
        with pytest.raises(FloatingPointError, match="non-finite logits"):
            graphs(*values, temperature=.7, top_k=4, top_p=.9)
