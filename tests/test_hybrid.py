"""BF16 loading regression and bounded native expert bridge checks."""
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gguf
import numpy as np
import pytest
import torch

from fish_drama3.hybrid import ExpertBridge, NativeExperts, check_gpu_budget, copy_parameter
import fish_drama3.hybrid as hybrid

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/drama3-direct-layer4/model.gguf"
DLL = ROOT / "builds/drama3-inspect/Release/drama3-experts-bridge.dll"
AVAILABLE = MODEL.is_file() and DLL.is_file()


def test_gpu_budget_reserves_backbone_before_bank_allocation(monkeypatch) -> None:
    tensors = [SimpleNamespace(name=f"model.layers.{layer}.mlp.experts_bank.gate_proj.weight",
                               data=SimpleNamespace(nbytes=size))
               for layer, size in ((0, 2000), (39, 1000))]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (9 * 2**30 + 1500, 16 * 2**30))
    reader = SimpleNamespace(tensors=tensors)
    check_gpu_budget(reader, 1)
    with pytest.raises(ValueError, match="reduce GPU layers"):
        check_gpu_budget(reader, 40)


def test_raw_bf16_parameter_copy_preserves_bits() -> None:
    values = torch.tensor([
        [0.0, -0.0, 1.0, -1.0, 0.333, -0.777, 123.5, 0.001],
        [2.0, -3.0, 0.5, 1.75, 0.0001, -0.0002, 15.125, -23.5],
        [1.01, 2.02, 3.03, 4.04, 5.05, 6.06, 7.07, 8.08],
    ], dtype=torch.bfloat16)
    raw = bytearray(values.view(torch.uint16).numpy().tobytes())
    tensor = SimpleNamespace(
        name="tower.test.weight", shape=np.array([8, 3]),
        tensor_type=gguf.GGMLQuantizationType.BF16,
        data=np.frombuffer(raw, dtype=np.uint8).reshape(3, 16),
    )
    target = torch.empty((3, 8), dtype=torch.bfloat16)
    copy_parameter(target, tensor, rows=2)
    assert torch.equal(target.view(torch.uint16), values.view(torch.uint16))


@pytest.fixture
def bridge() -> Iterator[ExpertBridge]:
    if not AVAILABLE:
        pytest.skip("Native bridge DLL or bounded layer artifact absent")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    instance = ExpertBridge(MODEL, threads=1)
    try:
        yield instance
    finally:
        instance.close()
        instance.close()
        torch.set_num_threads(previous)


def bridge_inputs(tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1969)
    values = torch.randn((tokens, 2048), generator=generator) * 0.02
    ids = torch.tensor([[0, 1, 2, 3, 10, 100, 254, 255]]).repeat(tokens, 1)
    if tokens > 1:
        ids[1] = torch.tensor([255, 254, 100, 10, 3, 2, 1, 0])
    mixing = torch.arange(1, 9, dtype=torch.float32).repeat(tokens, 1)
    mixing /= mixing.sum(dim=-1, keepdim=True)
    return values, ids, mixing


def independent_experts(values: torch.Tensor, ids: torch.Tensor,
                        mixing: torch.Tensor) -> torch.Tensor:
    reader = gguf.GGUFReader(str(MODEL))
    banks = {tensor.name.split(".")[-2]: tensor for tensor in reader.tensors}
    outputs = torch.zeros_like(values)
    for token in range(len(values)):
        for index, expert in enumerate(ids[token].tolist()):
            matrices = {
                name: torch.from_numpy(gguf.dequantize(
                    banks[name].data[expert], gguf.GGMLQuantizationType.Q4_K,
                ).copy())
                for name in ("gate_proj", "up_proj", "down_proj")
            }
            gate = matrices["gate_proj"] @ values[token]
            up = matrices["up_proj"] @ values[token]
            routed = matrices["down_proj"] @ (torch.nn.functional.silu(gate) * up)
            outputs[token] += mixing[token, index] * routed
    return outputs


@pytest.mark.parametrize("tokens", [1, 2])
def test_bridge_selected_expert_parity(bridge: ExpertBridge, tokens: int) -> None:
    values, ids, mixing = bridge_inputs(tokens)
    reference = independent_experts(values, ids, mixing)
    output = NativeExperts(bridge, 4)(values, ids, mixing)
    assert output.shape == values.shape and torch.isfinite(output).all()
    for token in range(tokens):
        error = output[token] - reference[token]
        relative = float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference[token]))
        maximum = float(error.abs().max())
        print(f"Bridge tokens={tokens} token={token}: relative RMS={relative:.9g}; max abs={maximum:.9g}")
        assert relative <= 0.02, f"relative RMS={relative}; max abs={maximum}"


def test_bridge_invalid_layer_and_expert(bridge: ExpertBridge) -> None:
    values, ids, mixing = bridge_inputs(1)
    with pytest.raises(RuntimeError, match="layer banks unavailable"):
        bridge.forward(3, values, ids, mixing)
    ids[0, 0] = 256
    with pytest.raises(RuntimeError, match="expert IDs"):
        bridge.forward(4, values, ids, mixing)


def test_closed_bridge_rejects_forward(bridge: ExpertBridge) -> None:
    values, ids, mixing = bridge_inputs(1)
    bridge.close()
    bridge.close()
    with pytest.raises(RuntimeError, match="closed"):
        bridge.forward(4, values, ids, mixing)


@pytest.mark.parametrize("bad_buffer", ["values", "ids", "mixing", "id_dtype"])
def test_bridge_rejects_unsafe_buffers(bridge: ExpertBridge, bad_buffer: str) -> None:
    values, ids, mixing = bridge_inputs(1)
    if bad_buffer == "values":
        values = values[:, :2047]
    elif bad_buffer == "ids":
        ids = ids[:, :7]
    elif bad_buffer == "mixing":
        mixing = mixing[:, :7]
    else:
        ids = ids.float()
    with pytest.raises(ValueError):
        bridge.forward(4, values, ids, mixing)


def test_auto_planner_uses_actual_tail_sizes_and_percentage() -> None:
    unit = 2**20
    sizes = [100 * unit, 300 * unit, 200 * unit]
    plan = hybrid.plan_gpu_layers(sizes, 900 * unit, 1000 * unit, 3)
    assert plan["headroom_bytes"] == 30 * unit
    assert plan["workspace_reserve_bytes"] == 256 * unit
    assert plan["gpu_expert_layers"] == 3
    assert plan["expert_bytes"] == 600 * unit
    plan = hybrid.plan_gpu_layers(sizes, 600 * unit, 1000 * unit, 3)
    assert plan["gpu_expert_layers"] == 1 and plan["expert_bytes"] == 200 * unit
    with pytest.raises(RuntimeError, match="headroom/workspace"):
        hybrid.plan_gpu_layers(sizes, 200 * unit, 1000 * unit, 3)


def test_expert_sizes_count_only_quantized_bank_payloads(monkeypatch) -> None:
    tensors = [SimpleNamespace(name=name, data=SimpleNamespace(nbytes=size)) for name, size in [
        ("model.layers.39.mlp.experts_bank.gate_proj.weight", 12),
        ("model.layers.39.mlp.experts_bank.up_proj.weight", 24),
        ("model.layers.0.mlp.experts_bank.down_proj.weight", 48),
        ("model.layers.39.attn.q_proj.weight", 999)]]
    monkeypatch.setattr(hybrid, "GGUFReader", lambda path: SimpleNamespace(tensors=tensors))
    sizes = hybrid.expert_layer_bytes(Path("model"))
    assert len(sizes) == 40 and sizes[39] == 36 and sizes[0] == 48
    assert sum(sizes) == 84


def fake_auto_model():
    layers = [SimpleNamespace(mlp=SimpleNamespace(experts=object())) for _ in range(40)]
    return SimpleNamespace(backbone=SimpleNamespace(model=SimpleNamespace(layers=layers)))


def test_auto_allocation_retries_and_rebinds_owned_bridge(monkeypatch) -> None:
    model, old, replacement = fake_auto_model(), Mock(), Mock()
    constructor = Mock(side_effect=[RuntimeError("GPU allocation failed"), replacement])
    monkeypatch.setattr(hybrid, "ExpertBridge", constructor)
    empty = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    placement = dict(gpu_expert_layers=3, headroom_bytes=512)
    result = hybrid.allocate_auto_experts(Path("model"), model, old, 8, placement, [10] * 40)
    assert result is replacement and placement["gpu_expert_layers"] == 2
    assert [call.args[2] for call in constructor.call_args_list] == [3, 2]
    assert constructor.call_args.kwargs["headroom_bytes"] == 512
    assert all(layer.mlp.experts.bridge is replacement for layer in model.backbone.model.layers)
    old.close.assert_called_once()
    replacement.close.assert_not_called()
    empty.assert_called_once()


def test_manual_allocation_failure_never_changes_requested_layers(monkeypatch) -> None:
    old = Mock()
    constructor = Mock(side_effect=RuntimeError("GPU allocation failed"))
    monkeypatch.setattr(hybrid, "ExpertBridge", constructor)
    with pytest.raises(RuntimeError, match="allocation failed"):
        hybrid.allocate_auto_experts(Path("model"), fake_auto_model(), old, 8,
                                     dict(gpu_expert_layers=16, headroom_bytes=512), [1] * 40,
                                     automatic=False)
    constructor.assert_called_once()
    old.close.assert_not_called()


def test_allocation_falls_back_to_existing_cpu_bridge(monkeypatch) -> None:
    old = Mock()
    monkeypatch.setattr(hybrid, "ExpertBridge", Mock(side_effect=RuntimeError("No VRAM")))
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    placement = dict(gpu_expert_layers=2, headroom_bytes=512)
    result = hybrid.allocate_auto_experts(Path("model"), fake_auto_model(), old, 8,
                                          placement, [1] * 40)
    assert result is old and placement["gpu_expert_layers"] == 0
    old.close.assert_not_called()


@pytest.mark.parametrize("manual", [None, 0, 16])
def test_staged_loading_plans_after_dense_weights_and_caches(monkeypatch, manual) -> None:
    model, old, replacement = fake_auto_model(), Mock(), Mock()
    sequence = []
    monkeypatch.setattr(hybrid, "load_model", lambda *args, **kwargs: (
        sequence.append(("dense", kwargs)) or (model, old)))
    monkeypatch.setattr(hybrid, "ensure_drama3_caches", lambda value, capacity: sequence.append(("cache", capacity)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: sequence.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 * 2**30, 16 * 2**30))
    monkeypatch.setattr(hybrid, "expert_layer_bytes", lambda bundle: [100 * 2**20] * 40)
    monkeypatch.setattr(hybrid, "ExpertBridge", Mock(return_value=replacement))
    _, bridge, placement = hybrid.load_model_placed(Path("model"), gpu_layers=manual)
    assert sequence[0] == ("dense", dict(threads=8, gpu_layers=0))
    assert sequence[1] == ("cache", 10240)
    assert placement["gpu_expert_layers"] == (40 if manual is None else manual)
    assert bridge is (old if manual == 0 else replacement)
    assert placement["headroom_bytes"] == int(np.ceil(16 * 2**30 * .03))
    assert placement["placement_mode"] == ("auto" if manual is None else "manual")


def test_manual_overbudget_closes_initial_cpu_bridge(monkeypatch) -> None:
    old = Mock()
    monkeypatch.setattr(hybrid, "load_model", lambda *args, **kwargs: (fake_auto_model(), old))
    monkeypatch.setattr(hybrid, "ensure_drama3_caches", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1024 * 2**20, 16 * 2**30))
    monkeypatch.setattr(hybrid, "expert_layer_bytes", lambda bundle: [100 * 2**20] * 40)
    with pytest.raises(ValueError, match="Requested GPU"):
        hybrid.load_model_placed(Path("model"), gpu_layers=16)
    old.close.assert_called_once()


def test_budget_native_abi_preserves_legacy_entry_point() -> None:
    bridge = object.__new__(ExpertBridge)
    budget, legacy = Mock(return_value=123), Mock(return_value=456)
    bridge.library = SimpleNamespace(drama3_open_cuda_budget=budget, drama3_open_cuda=legacy)
    import ctypes
    error = ctypes.create_string_buffer(1024)
    assert bridge.open_cuda(Path("model.gguf"), 16, 512, error) == 123
    assert budget.call_args.args[3] == 512
    assert budget.argtypes[3] is ctypes.c_uint64
    assert bridge.open_cuda(Path("model.gguf"), 16, None, error) == 456
    assert len(legacy.call_args.args) == 5


def test_missing_budget_abi_is_not_silently_retried_as_allocation_failure(monkeypatch):
    old = Mock()
    constructor = Mock(side_effect=ValueError("Auto placement requires updated Windows native libraries"))
    monkeypatch.setattr(hybrid, "ExpertBridge", constructor)
    with pytest.raises(ValueError, match="updated Windows native libraries"):
        hybrid.allocate_auto_experts(Path("model"), fake_auto_model(), old, 8,
                                     dict(gpu_expert_layers=16, headroom_bytes=512), [1] * 40)
    constructor.assert_called_once()
    old.close.assert_not_called()
