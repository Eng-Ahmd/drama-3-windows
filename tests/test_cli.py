"""Check completed audio publication and bounded reference preprocessing."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fish_drama3.cli import output_claim, publish_wave, reference_audio


@pytest.mark.parametrize("guarded_parent", [False, True])
def test_cli_respects_shared_lease_and_guarded_parent(tmp_path: Path, monkeypatch, guarded_parent: bool) -> None:
    import fish_drama3.cli as cli
    import fish_drama3.watchdog as watchdog

    lock = tmp_path / "gpu.lock"
    lease = watchdog.guarded_job_lock
    monkeypatch.setattr(watchdog, "guarded_job_lock", lambda: lease(lock))
    monkeypatch.setattr(watchdog, "start_child_telemetry", lambda: (Mock(), Mock()))
    monkeypatch.setattr(cli, "parse_options", lambda: SimpleNamespace(output=tmp_path / "result.wav"))
    synthesis = Mock(return_value=dict(complete=True))
    monkeypatch.setattr(cli, "synthesize", synthesis)
    monkeypatch.setenv("FISH_DRAMA3_WATCHDOG_ACTIVE", "1" if guarded_parent else "0")
    with lease(lock):
        if guarded_parent:
            cli.main()
            synthesis.assert_called_once()
        else:
            with pytest.raises(SystemExit) as error:
                cli.main()
            assert error.value.code == 2
            synthesis.assert_not_called()
    assert not (tmp_path / "result.lock").exists()


def test_concurrent_claim_refused_and_released(tmp_path: Path) -> None:
    path = tmp_path / "result.wav"
    with output_claim(path):
        with pytest.raises(FileExistsError, match="claimed"):
            with output_claim(path):
                pass
    assert not path.with_suffix(".lock").exists()


def test_publication_preserves_late_existing_output(tmp_path: Path) -> None:
    path = tmp_path / "result.wav"
    path.write_bytes(b"previous result")
    with pytest.raises(FileExistsError):
        publish_wave(path, np.zeros(8, dtype=np.float32), dict(complete=True))
    assert path.read_bytes() == b"previous result"
    assert not list(tmp_path.glob("*.partial"))


def test_completed_pcm_wave_and_report_are_readable(tmp_path: Path) -> None:
    path = tmp_path / "result.wav"
    audio = np.linspace(-0.5, 0.5, 1024, dtype=np.float32)
    publish_wave(path, audio, dict(complete=True, frames=1))
    values, rate = sf.read(path, dtype="float32")
    assert rate == 44100 and values.shape == audio.shape
    np.testing.assert_allclose(values, audio, rtol=0, atol=1 / 32768)
    assert json.loads(path.with_suffix(".json").read_text())["complete"] is True
    assert not list(tmp_path.glob("*.partial"))


def test_reference_mono_resampling_records_recipe(tmp_path: Path) -> None:
    path = tmp_path / "reference.wav"
    sf.write(path, np.zeros((2400, 2), dtype=np.float32), 24000, subtype="FLOAT")
    audio, record = reference_audio(path)
    assert audio.shape == (4410,) and audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert record["source_rate"] == 24000 and record["target_rate"] == 44100
    assert record["resampler"] == "scipy.signal.resample_poly/default"


def test_reference_rejects_oversized_header_before_read(tmp_path: Path) -> None:
    path = tmp_path / "too-many-channels.wav"
    sf.write(path, np.zeros((1, 9), dtype=np.float32), 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="channels"):
        reference_audio(path)


@pytest.mark.parametrize("allow_partial", [False, True])
def test_synthesis_partial_output_contract(tmp_path: Path, monkeypatch, allow_partial: bool) -> None:
    """A length-limited result must not silently become completed audio."""
    import sys
    import fish_drama3.cli as cli

    path = tmp_path / "partial.wav"
    args = SimpleNamespace(output=path, text="First. Second.", reference_text=None,
        reference_audio=None, bundle=tmp_path, codec=tmp_path / "codec.gguf",
        threads=1, seed=123, gpu_expert_layers=0, allow_partial=allow_partial,
        temperature=0.7, top_p=0.7, top_k=30, chunk_length=200)
    segment = dict(text="First.", frames=1, finish_reason="length")
    frames = MagicMock()
    frames.__len__.return_value = 1
    frames.numpy.return_value = np.zeros((1, 11), dtype=np.int64)
    codec, bridge = Mock(), Mock()
    codec.decode.return_value = np.linspace(-0.1, 0.1, 2048, dtype=np.float32)
    context = MagicMock()
    context.__enter__.return_value = codec
    monkeypatch.setattr(cli, "CodecBridge", Mock(return_value=context))
    monkeypatch.setattr(cli, "verify_assets", Mock(return_value={"verified": True}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=Mock(),
        manual_seed=Mock(), cuda=SimpleNamespace(max_memory_allocated=Mock(return_value=0))))
    monkeypatch.setitem(sys.modules, "fish_drama3.hybrid", SimpleNamespace(
        load_model=Mock(return_value=(Mock(), bridge))))
    monkeypatch.setitem(sys.modules, "fish_drama3.generation", SimpleNamespace(
        validate_drama3_text=Mock(), generate_text=Mock(return_value=(frames, "length", [segment]))))
    with output_claim(path):
        if allow_partial:
            result = cli.synthesize(args)
            report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert result["complete"] is False and report["complete"] is False
            assert report["finish_reason"] == "length" and report["segments"] == [segment]
            values, rate = sf.read(path)
            assert values.shape == (2048,) and rate == 44100
        else:
            with pytest.raises(RuntimeError, match="no complete WAV published"):
                cli.synthesize(args)
            codec.decode.assert_not_called()
    bridge.close.assert_called_once()
    context.__exit__.assert_called_once()
    assert not path.with_suffix(".lock").exists()
    assert not list(tmp_path.glob("*.partial"))
    if not allow_partial:
        assert not path.exists() and not path.with_suffix(".json").exists()
