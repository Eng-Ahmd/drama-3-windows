"""CPU contracts for the persistent browser worker protocol."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import fish_drama3.web_worker as worker


def request(tmp_path: Path, **changes: object) -> dict:
    return dict(text="Hello.", output=str(tmp_path / "result.wav"), **changes)


@pytest.mark.parametrize("changes", [
    {"max_frames": 8193}, {"gpu_expert_layers": 99, "top_p": 0},
    {"chunk_length": 12}, {"temperature": float("nan")},
    {"allow_partial": "false"}, {"seed": True},
    {"reference_text": "Missing audio"}, {"output_format": "flac"},
])
def test_generation_validation(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        worker.generation_options(request(tmp_path, **changes))


def test_options_defaults_and_replay_metadata(tmp_path: Path) -> None:
    payload = request(tmp_path, output_format="mp3", mp3_bitrate=192)
    args = worker.generation_options(payload)
    assert args.max_frames == 512 and args.seed == 123
    session = worker.Session(dict(bundle="model", codec="codec", threads=8, gpu_expert_layers=12))
    session.load_seconds = 10.0
    result = session.metrics(payload, args, "length", [], None)
    assert result["complete"] is False
    assert result["settings"]["output_format"] == "mp3"
    assert result["settings"]["max_frames"] == 512
    assert result["settings"]["output"] == str(tmp_path / "result.wav")
    assert result["settings"]["bundle"] == "model"


def test_requests_recover_from_validation_and_reuse_session(monkeypatch, capsys) -> None:
    session = Mock()
    session.generate.side_effect = [ValueError("Bad prompt"), {"output": "result.wav"}]
    commands = [{"action": "generate"}, {"action": "generate"}, {"action": "unload"}]
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("\n".join(map(json.dumps, commands))))
    worker.handle_requests(session)
    events = [json.loads(line.removeprefix("DRAMA3_EVENT"))
              for line in capsys.readouterr().out.splitlines()]
    assert events[0]["status"] == "ready" and events[0]["error"] == "Bad prompt"
    assert events[1]["result"]["output"] == "result.wav"
    assert session.generate.call_count == 2


def test_cuda_failure_stops_worker(monkeypatch, capsys) -> None:
    session = Mock()
    session.generate.side_effect = RuntimeError("CUDA out of memory")
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO('{"action":"generate"}\n'))
    with pytest.raises(RuntimeError, match="CUDA"):
        worker.handle_requests(session)
    event = json.loads(capsys.readouterr().out.removeprefix("DRAMA3_EVENT"))
    assert event["status"] == "error"


def test_close_releases_codec_even_if_bridge_fails() -> None:
    session = worker.Session(dict(bundle="model", codec="codec"))
    bridge, codec = Mock(), Mock()
    session.bridge, session.codec = bridge, codec
    bridge.close.side_effect = RuntimeError("close failure")
    with pytest.raises(RuntimeError):
        session.close()
    codec.close.assert_called_once()
    assert session.model is session.bridge is session.codec is None


def test_preflight_checks_existing_safety_limits(monkeypatch) -> None:
    import sys

    monkeypatch.delenv("FISH_DRAMA3_WEB_HEADROOM_PERCENT", raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda device: (256 * 2**20, 16 * 2**30))))
    monkeypatch.setattr(worker, "sample_memory", lambda: dict(
        gpu_free_bytes=256 * 2**20, ram_free_bytes=32 * 2**30, commit_percent=50))
    with pytest.raises(RuntimeError, match="gpu_headroom"):
        worker.preflight()


@pytest.mark.parametrize("allow_partial", [False, True])
def test_generation_uses_loaded_resources_and_marks_partial(tmp_path, monkeypatch, allow_partial):
    import sys
    import numpy as np

    session = worker.Session(dict(bundle="model", codec="codec"))
    session.model, session.codec, session.bridge = Mock(), Mock(), Mock()
    session.load_seconds = 3.0
    session.codec.decode.return_value = np.array([.1, -.1], dtype=np.float32)
    frames = type("Frames", (), {"__len__": lambda self: 1,
                                 "numpy": lambda self: np.zeros((1, 11))})()
    generation = Mock(return_value=(frames, "length", []))
    from contextlib import nullcontext
    torch = SimpleNamespace(manual_seed=Mock(), inference_mode=nullcontext,
                            cuda=SimpleNamespace(reset_peak_memory_stats=Mock(),
                                                 max_memory_allocated=Mock(return_value=0)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "fish_drama3.generation", SimpleNamespace(
        generate_text=generation, validate_drama3_text=Mock()))
    monkeypatch.setattr(worker, "preflight", Mock())
    payload = request(tmp_path, allow_partial=allow_partial)
    if allow_partial:
        result = session.generate(payload)
        assert result["metrics"]["complete"] is False
        sidecar = json.loads((tmp_path / "result.json").read_text())
        assert sidecar["settings"]["max_frames"] == 512
        assert sidecar["settings"]["allow_partial"] is True
    else:
        with pytest.raises(RuntimeError, match="Frame limit"):
            session.generate(payload)
        session.codec.decode.assert_not_called()
        assert not (tmp_path / "result.wav").exists()
    assert generation.call_args.args[0] is session.model
    torch.manual_seed.assert_called_once_with(123)
    session.bridge.close.assert_not_called()


def test_lifetime_monitor_uses_actual_cuda_headroom(monkeypatch, capsys):
    import sys
    import threading

    monkeypatch.delenv("FISH_DRAMA3_WEB_HEADROOM_PERCENT", raising=False)
    monkeypatch.setattr(worker, "sample_memory", lambda: dict(
        gpu_free_bytes=8 * 2**30, ram_free_bytes=32 * 2**30, commit_percent=50))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda device: (256 * 2**20, 16 * 2**30))))

    def terminate(code):
        raise SystemExit(code)

    monkeypatch.setattr(worker.os, "_exit", terminate)
    with pytest.raises(SystemExit) as error:
        worker.monitor_memory(threading.Event())
    assert error.value.code == 2
    event = json.loads(capsys.readouterr().out.removeprefix("DRAMA3_EVENT"))
    assert event["status"] == "error" and "gpu_headroom" in event["error"]


def test_full_unchunked_podcast_budget_is_accepted(tmp_path):
    args = worker.generation_options(request(tmp_path, max_frames=6144, chunk_length=0))
    assert args.max_frames == 6144 and args.chunk_length == 0


def test_memory_headroom_defaults_to_three_percent(monkeypatch):
    monkeypatch.delenv("FISH_DRAMA3_WEB_HEADROOM_PERCENT", raising=False)
    assert worker.memory_limits(10000).gpu_free_bytes == 300
    monkeypatch.setenv("FISH_DRAMA3_WEB_HEADROOM_PERCENT", "5")
    assert worker.memory_limits(10000).gpu_free_bytes == 500


@pytest.mark.parametrize("mode,layers", [("auto", None), ("manual", 40)])
def test_session_load_uses_staged_placement_helper(monkeypatch, mode, layers):
    import sys

    model, bridge, codec = Mock(), Mock(), Mock()
    placement = dict(gpu_expert_layers=24 if mode == "auto" else 40)
    load = Mock(return_value=(model, bridge, placement))
    monkeypatch.setitem(sys.modules, "fish_drama3.hybrid", SimpleNamespace(load_model_placed=load))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=Mock()))
    monkeypatch.setattr(worker, "preflight", Mock())
    monkeypatch.setattr(worker, "verify_assets", Mock(return_value={"verified": True}))
    monkeypatch.setattr(worker, "CodecBridge", Mock(return_value=codec))
    session = worker.Session(dict(bundle="model", codec="codec", gpu_expert_layers=40,
                                  placement_mode=mode, headroom_percent=3))
    session.load()
    load.assert_called_once_with(Path("model"), threads=8, gpu_layers=layers,
                                 headroom_percent=3, cache_capacity=10240)
    assert session.model is model and session.bridge is bridge and session.codec is codec
    assert session.gpu_layers == placement["gpu_expert_layers"]


@pytest.mark.parametrize("frame_count", [2048, 2049, 3584, 6144])
def test_long_decode_preserves_continuous_samples_across_context_boundaries(frame_count):
    import numpy as np

    session = worker.Session(dict(bundle="model", codec="codec"))
    calls = []

    def decode(rows, threads):
        first, count = int(rows[0, 0]), len(rows)
        calls.append((first, count))
        assert count <= 2048 and threads == session.threads
        return np.arange(first * 2048, (first + count) * 2048, dtype=np.float32)

    session.codec = SimpleNamespace(decode=decode)
    rows = np.arange(frame_count, dtype=np.int64).reshape(-1, 1)
    audio = session.decode_audio(rows)
    np.testing.assert_array_equal(audio, np.arange(frame_count * 2048, dtype=np.float32))
    if frame_count == 2048:
        assert calls == [(0, frame_count)]
    else:
        assert calls[0] == (0, 1792)
        for index, (first, count) in enumerate(calls[1:], start=1):
            assert first == index * 1792 - 256
            assert count == min(1792 + 256, frame_count - first)
def test_passage_progress_distinguishes_prefill_from_audio_generation(capsys):
    progress = dict(segment_frames=0, segment_complete=False,
                    segment_index=2, segment_count=4)
    worker.emit_generation_progress(progress)
    event = json.loads(capsys.readouterr().out.removeprefix("DRAMA3_EVENT"))
    assert event["phase"] == "prefill" and "2 of 4" in event["message"]
    worker.emit_generation_progress(dict(progress, segment_frames=7))
    event = json.loads(capsys.readouterr().out.removeprefix("DRAMA3_EVENT"))
    assert event["phase"] == "generating"
