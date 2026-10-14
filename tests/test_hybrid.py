"""BF16 loading regression and bounded native expert bridge checks."""
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from fish_drama3.hybrid import ExpertBridge, NativeExperts, check_gpu_budget, copy_parameter

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
