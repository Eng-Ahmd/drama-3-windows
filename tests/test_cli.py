"""Check completed audio publication and bounded reference preprocessing."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
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
