"""Synthetic checkpoint planning checks without loading model weights."""
import copy

import pytest

from fish_drama3.direct import TensorPlan, choose_storage, plan_tensors
from fish_drama3.native_quant import Q4_K, Q8_0


def expert_headers() -> dict:
    """Intentionally insert expert keys in string-sort order."""
    headers = {}
    for expert in sorted(range(256), key=str):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = [2048, 512] if projection == "down_proj" else [512, 2048]
            name = f"model.language_model.layers.4.mlp.experts.{expert}.{projection}.weight"
            headers[name] = {"dtype": "F8_E4M3", "shape": shape}
            headers[name + "_scale_inv"] = {
                "dtype": "F32", "shape": [dimension // 128 for dimension in shape],
            }
    return headers


def test_experts_numeric_order_geometry_and_byte_budget() -> None:
    plans = plan_tensors(expert_headers(), {4}, experts_only=True)
    assert len(plans) == 3
    for plan in plans:
        assert plan.storage == Q4_K
        assert [int(name.split(".experts.")[1].split(".")[0])
                for name in plan.sources] == list(range(256))
        assert plan.nbytes == 256 * 512 * 2048 // 256 * 144
        if "down_proj" in plan.name:
            assert plan.shape == (256, 2048, 512)
        else:
            assert plan.shape == (256, 512, 2048)


@pytest.mark.parametrize("suffix", ["", "_scale_inv"])
def test_missing_expert_or_scale_fails(suffix: str) -> None:
    headers = expert_headers()
    del headers["model.language_model.layers.4.mlp.experts.10.gate_proj.weight" + suffix]
    with pytest.raises(ValueError, match="Incomplete bank|Missing/invalid scale"):
        plan_tensors(headers, {4}, experts_only=True)


@pytest.mark.parametrize("field,value", [("shape", [512, 512]), ("dtype", "BF16")])
def test_wrong_expert_contract_fails(field: str, value: object) -> None:
    headers = expert_headers()
    headers["model.language_model.layers.4.mlp.experts.0.down_proj.weight"][field] = value
    with pytest.raises(ValueError, match="Unsupported expert geometry"):
        plan_tensors(headers, {4}, experts_only=True)


def test_down_scale_axes_are_not_gate_axes() -> None:
    headers = expert_headers()
    headers["model.language_model.layers.4.mlp.experts.0.down_proj.weight_scale_inv"]["shape"] = [4, 16]
    with pytest.raises(ValueError, match="Missing/invalid scale"):
        plan_tensors(headers, {4}, experts_only=True)


@pytest.mark.parametrize("name,shape,dtype,storage", [
    ("model.language_model.layers.4.mlp.gate.weight", [256, 2048], "BF16", 0),
    ("model.language_model.layers.4.mlp.shared_expert_gate.weight", [1, 2048], "BF16", 0),
    ("model.language_model.layers.4.input_layernorm.weight", [2048], "BF16", 0),
    ("model.language_model.layers.4.linear_attn.A_log", [32], "F32", 0),
    ("model.language_model.layers.4.self_attn.q_proj.weight", [2048, 2048], "BF16", Q8_0),
    ("audio_decoder_mtp.heads.0.weight", [1024, 2048], "BF16", 30),
    ("audio_decoder_mtp_shared.layers.0.norm.weight", [2048], "BF16", 0),
])
def test_sensitive_and_tower_storage(name: str, shape: list[int], dtype: str,
                                     storage: int) -> None:
    assert choose_storage(name, {"shape": shape, "dtype": dtype}) == storage


def test_destination_mapping_and_component_selection() -> None:
    headers = expert_headers()
    dense = {
        "audio_decoder.codebook_embeddings.0.weight": {"dtype": "BF16", "shape": [1024, 2048]},
        "model.language_model.embed_tokens.weight": {"dtype": "BF16", "shape": [252288, 2048]},
    }
    headers.update(copy.deepcopy(dense))
    full = plan_tensors(headers, {4}, experts_only=False)
    assert len(full) == 5
    assert any(plan.name == "codebook_embeddings.0.weight" for plan in full)
    assert len(plan_tensors(headers, {4}, experts_only=True)) == 3


def test_unaligned_quantized_plan_fails() -> None:
    plan = TensorPlan("bad", (2, 257), Q4_K, ("source",))
    with pytest.raises(ValueError, match="Unaligned"):
        _ = plan.nbytes
