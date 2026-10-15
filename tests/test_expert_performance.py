"""Parity and concurrent telemetry checks for optional native expert optimizations."""
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
import ctypes
import os
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from fish_drama3.hybrid import ExpertBridge
from fish_drama3.watchdog import guarded_job_lock

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/drama3-direct-layer4/model.gguf"
CANDIDATE = Path(os.environ.get("FISH_DRAMA3_TEST_EXPERT_CPU_DLL", str(
    ROOT / "builds/drama3-cpu-perf/Release/drama3-experts-bridge.dll")))
CUDA_CANDIDATE = Path(os.environ.get("FISH_DRAMA3_TEST_EXPERT_CUDA_DLL", str(
    ROOT / "builds/drama3-cuda-perf/Release/drama3-experts-bridge.dll")))
FULL_MODEL = ROOT / "models/drama3-hub-compatible/model.gguf"


@pytest.fixture(scope="module", autouse=True)
def production_dependency_order() -> Iterator[ctypes.CDLL | None]:
    """Optionally mirror the studio's codec-before-experts Windows DLL order."""
    codec_library = os.environ.get("FISH_DRAMA3_TEST_CODEC_DLL")
    if not codec_library:
        yield None
        return
    with os.add_dll_directory(str(Path(codec_library).resolve().parent)):
        library = ctypes.CDLL(str(Path(codec_library).resolve()))
        yield library


@pytest.fixture
def cuda_expert_lease() -> Iterator[None]:
    with guarded_job_lock():
        yield


def test_automatic_cuda_device_dispatch_only_uses_measured_prefill_shape() -> None:
    bridge = object.__new__(ExpertBridge)
    bridge.device_enabled, bridge.device_min_tokens, bridge.gpu_layers = True, 128, 16
    tensor = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"), shape=(128, 2048))
    assert bridge.uses_device_bridge(24, tensor)
    assert not bridge.uses_device_bridge(23, tensor)
    tensor.shape = (1, 2048)
    assert not bridge.uses_device_bridge(24, tensor)
    bridge.device_min_tokens = 1
    assert bridge.uses_device_bridge(24, tensor)
    tensor.device = torch.device("cuda:1")
    assert not bridge.uses_device_bridge(24, tensor)
    tensor.device, tensor.is_cuda = torch.device("cpu"), False
    assert not bridge.uses_device_bridge(24, tensor)


def test_profile_snapshots_are_independent_and_preserve_concurrent_updates() -> None:
    bridge = object.__new__(ExpertBridge)
    bridge.profile_enabled = True
    bridge.profile_totals = {}
    bridge.profile_lock = Lock()
    bridge.performance = {}
    bridge.gpu_layers = 1

    def record(layer: int) -> None:
        for _ in range(2000):
            bridge.record_profile(layer, 1, 1.0, 1.1, 1.2, 1.3)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(record, layer) for layer in (4, 39)]
        snapshots = [bridge.profile_summary(reset=True) for _ in range(100)]
        for future in futures:
            future.result()
    snapshots.append(bridge.profile_summary())
    for backend in ("cpu", "gpu"):
        assert sum(item["backends"].get(backend, {}).get("calls", 0)
                   for item in snapshots) == 2000
    bridge.profile_summary(reset=True)
    bridge.record_profile(4, 1, 1.0, 1.1, 1.2, 1.3)
    snapshot = bridge.profile_summary()
    snapshot["backends"].clear()
    assert bridge.profile_summary()["backends"]["cpu"]["calls"] == 1


@pytest.mark.skipif(not MODEL.is_file() or not CANDIDATE.is_file(),
                    reason="Separate candidate DLL or bounded expert fixture absent")
def test_cached_decode_graph_matches_uncached_and_validates_inputs() -> None:
    bridge = ExpertBridge(MODEL, threads=2, library=CANDIDATE, profile=True)
    generator = torch.Generator().manual_seed(1919)
    values = torch.randn((1, 2048), generator=generator) * 0.02
    ids = torch.tensor([[0, 1, 2, 3, 10, 100, 254, 255]])
    weights = torch.full((1, 8), 0.125)
    try:
        bridge.configure_performance(poll=50, cache=False)
        reference = bridge.forward(4, values, ids, weights)
        bridge.configure_performance(poll=0, cache=True)
        for _ in range(3):
            actual = bridge.forward(4, values, ids, weights)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        values[0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="nonfinite"):
            bridge.forward(4, values, ids, weights)
        values[0, 0] = 0
        ids[0, 0] = 256
        with pytest.raises(RuntimeError, match="expert IDs"):
            bridge.forward(4, values, ids, weights)
    finally:
        bridge.close()


@pytest.mark.skipif(os.environ.get("FISH_DRAMA3_TEST_CUDA_EXPERTS") != "1",
                    reason="Opt-in CUDA expert allocation regression")
@pytest.mark.parametrize("tokens", [1, 128])
def test_cuda_device_bf16_parity_and_rejects_host_pointer(
        monkeypatch, cuda_expert_lease, tokens: int) -> None:
    monkeypatch.setenv("FISH_DRAMA3_GPU_DEVICE_BRIDGE", "1")
    bridge = ExpertBridge(FULL_MODEL, threads=2, gpu_layers=1, library=CUDA_CANDIDATE)
    values = torch.randn((tokens, 2048), device="cuda:0", dtype=torch.bfloat16) * 0.02
    ids = torch.arange(8, device="cuda:0").reshape(1, 8).repeat(tokens, 1)
    weights = torch.full((tokens, 8), 0.125, device="cuda:0")
    try:
        bridge.device_enabled = False
        reference = bridge.forward(39, values, ids, weights)
        bridge.device_enabled = True
        torch.testing.assert_close(bridge.forward(39, values, ids, weights), reference,
                                   rtol=0, atol=0)
        values[0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="nonfinite CUDA"):
            bridge.forward(39, values, ids, weights)
        host = torch.zeros((1, 2048), dtype=torch.float32)
        host_ids = ids.int().cpu().contiguous()
        host_weights = weights.cpu().contiguous()
        error = ctypes.create_string_buffer(1024)
        status = bridge.device_entry(bridge.handle, 39, 1, host.data_ptr(),
            host_ids.data_ptr(), host_weights.data_ptr(), host.data_ptr(), 2, error, len(error))
        assert status != 0 and error.value
        padded = torch.zeros(2050, device="cuda:0", dtype=torch.float32)
        for x_pointer, out_pointer in ((padded.data_ptr() + 1, padded.data_ptr()),
                                       (padded.data_ptr(), padded.data_ptr() + 1)):
            status = bridge.device_entry(bridge.handle, 39, 1, x_pointer,
                host_ids.data_ptr(), host_weights.data_ptr(), out_pointer, 2, error, len(error))
            assert status != 0 and b"aligned float buffers" in error.value
    finally:
        bridge.close()
