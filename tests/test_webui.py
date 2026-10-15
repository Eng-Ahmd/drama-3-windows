"""CPU tests for studio validation, worker ownership, and localhost HTTP."""
from __future__ import annotations

from http.client import HTTPConnection
import io
import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from fish_drama3 import web_config, web_service, webui

MODEL = dict(bundle="model", codec="codec", threads=8, gpu_expert_layers=12)


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(web_service, "find_encoder", lambda: None)
    return web_service.Studio(tmp_path / "output")


@pytest.fixture
def server(studio):
    instance = webui.StudioServer(("127.0.0.1", 0), studio)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=5)


def http(server, path, method="GET", body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


@pytest.mark.parametrize("key,value", [
    ("seed", -1), ("seed", 2**32), ("seed", True), ("top_k", 0),
    ("max_frames", 8193), ("chunk_length", 31), ("chunk_length", 2001),
    ("temperature", float("inf")), ("temperature", 0), ("top_p", 1.1),
    ("allow_partial", "false"), ("mp3_bitrate", 64), ("text", " "),
])
def test_generation_bounds(key, value):
    with pytest.raises(ValueError):
        web_config.generation_options(dict(text="Hello", **{key: value})
                                      if key != "text" else {key: value}, True)


def test_mp3_requires_encoder_and_wav_keeps_replay_defaults():
    with pytest.raises(ValueError, match="FFmpeg"):
        web_config.generation_options(dict(text="Hello", output_format="mp3"), False)
    settings = web_config.generation_options(dict(text="Hello"), False)
    assert settings["seed"] == 123 and settings["max_frames"] == 512
    assert settings["output_format"] == "wav" and settings["allow_partial"] is False


def test_model_options_require_actual_bundle_and_bounds(tmp_path):
    bundle, codec = tmp_path / "model", tmp_path / "codec"
    bundle.mkdir()
    codec.mkdir()
    for path in (bundle / "model.gguf", bundle / "manifest.json",
                 codec / "codec.gguf", codec / "manifest.json"):
        path.write_text("fixture")
    data = dict(bundle=str(bundle), codec=str(codec / "codec.gguf"))
    assert web_config.model_options(data)["gpu_expert_layers"] == 12
    assert web_config.model_options(data)["placement_mode"] == "auto"
    with pytest.raises(ValueError):
        web_config.model_options(dict(data, gpu_expert_layers=41))
    with pytest.raises(ValueError):
        web_config.model_options(dict(data, threads=True))


def fake_worker(lines=""):
    process = Mock()
    process.stdin, process.stdout = io.StringIO(), io.StringIO(lines)
    process.poll.return_value = None
    process.wait.return_value = 0
    return process


def test_load_starts_one_persistent_worker_then_unload(studio, monkeypatch):
    process = fake_worker()
    monkeypatch.setattr(web_service, "model_options", lambda data: dict(bundle="model", codec="codec"))
    spawn = Mock(return_value=process)
    monkeypatch.setattr(web_service.subprocess, "Popen", spawn)
    monkeypatch.setattr(web_service.threading, "Thread", Mock())
    assert studio.load({})["status"] == "loading"
    assert json.loads(process.stdin.getvalue())["action"] == "load"
    with pytest.raises(ValueError, match="Unload"):
        studio.load({})
    assert studio.stop()["status"] == "unloaded"
    process.terminate.assert_called_once()
    assert studio.worker is None and studio.pending is None
    spawn.return_value = fake_worker()
    studio.load({})
    assert spawn.call_count == 2


def test_cancel_kills_worker_if_termination_times_out(studio):
    process = fake_worker()
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 15), 0]
    studio.worker = process
    result = studio.stop("Cancelled")
    process.kill.assert_called_once()
    assert result["status"] == "unloaded" and result["message"] == "Cancelled"


def test_generate_uses_registered_reference_and_unique_output(studio):
    studio.worker = fake_worker()
    studio.state.update(status="ready", model=dict(MODEL))
    reference = studio.output / "reference.wav"
    studio.references["voice"] = reference
    studio.generate(dict(text="Hello", reference_id="voice", reference_text="Source words"))
    payload = json.loads(studio.worker.stdin.getvalue())
    assert payload["reference_audio"] == str(reference)
    assert Path(payload["output"]).parent.parent == studio.output
    assert studio.snapshot()["status"] == "generating"
    with pytest.raises(ValueError, match="ready"):
        studio.generate(dict(text="Again"))


def test_publication_saves_settings_and_confines_worker_output(studio):
    studio.worker = fake_worker()
    studio.state.update(status="ready", model=dict(MODEL))
    studio.generate(dict(text="Hello", seed=77))
    identifier = studio.pending["id"]
    wav = studio.output / identifier / "speech.wav"
    wav.write_bytes(b"wave")
    wav.with_suffix(".json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        studio.prepare_result(dict(output=str(studio.output / "outside.wav"), metrics={}), studio.pending)
    result = studio.prepare_result(dict(output=str(wav), metrics=dict(complete=True)), studio.pending)
    settings = json.loads(wav.with_name("settings.json").read_text())
    assert settings["seed"] == 77 and settings["model"]["threads"] == 8
    assert settings["schema"] == "fish-drama3-studio-v1"
    assert result["wav_url"] in studio.exports


def test_mp3_failure_keeps_wav_and_removes_partial(studio, monkeypatch):
    wav = studio.output / "test.wav"
    wav.write_bytes(b"wave")
    wav.with_suffix(".mp3").write_bytes(b"partial")
    monkeypatch.setattr(web_service.subprocess, "run", Mock(side_effect=OSError("encoder failed")))
    audio, warning = studio.encode_mp3(wav, 192)
    assert audio == wav and "original WAV" in warning
    assert wav.exists() and not wav.with_suffix(".mp3").exists()


def test_reader_ignores_native_logs_and_stale_worker_events(studio):
    process = fake_worker('native log\nDRAMA3_EVENT{"status":"ready","message":"loaded"}\n')
    studio.worker = process
    studio.read_worker(process)
    assert studio.worker is None and studio.state["status"] == "error"
    assert "loaded" in studio.state["message"]
    studio.state.update(status="unloaded", message="new lifecycle")
    stale = fake_worker('DRAMA3_EVENT{"status":"ready"}\n')
    studio.read_worker(stale)
    assert studio.state["status"] == "unloaded"


@pytest.mark.parametrize("headers", [
    {"Host": "attacker.test"}, {"Origin": "https://attacker.test"},
    {"Origin": "null"},
])
def test_http_rejects_untrusted_host_or_origin(server, headers):
    status, _, body = http(server, "/api/state", headers=headers)
    assert status == 403 and "localhost" in json.loads(body)["error"]


def test_http_state_static_security_and_path_confinement(server):
    status, headers, body = http(server, "/api/state")
    assert status == 200 and json.loads(body)["status"] == "unloaded"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert http(server, "/assets/../../pyproject.toml")[0] == 404
    assert http(server, "/outputs/../../pyproject.toml")[0] == 404
    assert http(server, "/")[0] == 200


def test_http_download_ranges_only_registered_exports(server, studio):
    path = studio.output / "audio.wav"
    path.write_bytes(b"0123456789")
    studio.exports["/outputs/audio.wav"] = path
    status, headers, body = http(server, "/outputs/audio.wav", headers={"Range": "bytes=2-5"})
    assert status == 206 and body == b"2345"
    assert headers["Content-Range"] == "bytes 2-5/10"
    assert http(server, "/outputs/audio.wav", headers={"Range": "bytes=20-"})[0] == 416
    assert http(server, "/outputs/audio.wav", headers={"Range": "bytes=-5"})[0] == 416
    assert http(server, "/outputs/unregistered.wav")[0] == 404


@pytest.mark.parametrize("body,headers", [
    (b"[]", {"Content-Type": "application/json"}),
    (b"{", {"Content-Type": "application/json"}),
    (b"{}", {"Content-Type": "text/plain"}),
    (b"", {"Content-Type": "application/json"}),
])
def test_http_invalid_requests_return_json_errors(server, body, headers):
    status, _, response = http(server, "/api/generate", "POST", body, headers)
    assert status == 400 and "error" in json.loads(response)


def test_upload_audio_is_validated_and_filename_cannot_escape(server, studio):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(4410), 44100, format="WAV")
    status, _, response = http(server, "/api/reference", "POST", buffer.getvalue(),
                               {"X-Filename": "../../escape.wav"})
    assert status == 200
    record = json.loads(response)
    path = studio.references[record["reference_id"]]
    assert path.parent == studio.output / "references" and path.name != "escape.wav"
    before = set(path.parent.iterdir())
    assert http(server, "/api/reference", "POST", b"not audio", {"X-Filename": "bad.wav"})[0] == 400
    assert set(path.parent.iterdir()) == before
    assert http(server, "/api/reference", "POST", b"x", {"X-Filename": "bad.exe"})[0] == 400


def test_upload_rejects_large_content_length_before_read(server):
    status, _, response = http(server, "/api/reference", "POST", b"x", {
        "X-Filename": "big.wav", "Content-Length": str(48 * 1024 * 1024 + 1)})
    assert status == 400 and "bytes" in json.loads(response)["error"]


@pytest.mark.parametrize("error", [OSError("missing binary"), subprocess.TimeoutExpired("ffmpeg", 10)])
def test_encoder_detection_failure_disables_mp3(monkeypatch, error):
    monkeypatch.setattr(web_service.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(web_service.subprocess, "run", Mock(side_effect=error))
    assert web_service.find_encoder() is None


def test_worker_exit_before_generation_does_not_leave_busy_state(studio):
    studio.worker = fake_worker()
    studio.worker.poll.return_value = 2
    studio.state.update(status="ready", model=dict(MODEL))
    with pytest.raises(RuntimeError, match="stopped"):
        studio.generate(dict(text="Hello"))
    assert studio.pending is None and studio.state["status"] == "error"


def test_export_does_not_block_cancellation(studio, monkeypatch):
    process = fake_worker('DRAMA3_EVENT{"status":"ready","result":{"output":"result.wav"}}\n')
    studio.worker = process
    studio.pending = dict(id="result")
    entered, release = threading.Event(), threading.Event()

    def prepare(result, pending):
        entered.set()
        release.wait(5)
        return {"audio_url": "stale-output"}

    monkeypatch.setattr(studio, "prepare_result", prepare)
    reader = threading.Thread(target=studio.read_worker, args=(process,))
    reader.start()
    try:
        assert entered.wait(2)
        assert studio.stop("Cancelled")["status"] == "unloaded"
    finally:
        release.set()
        reader.join(3)
    assert studio.state["result"] is None


def test_sample_reference_rejects_unknown_and_hash_mismatch(server, monkeypatch, studio):
    from urllib import request

    payload = json.dumps(dict(id="unknown")).encode()
    headers = {"Content-Type": "application/json"}
    assert http(server, "/api/sample-reference", "POST", payload, headers)[0] == 400
    fetch = Mock(return_value=io.BytesIO(b"unexpected audio"))
    monkeypatch.setattr(request, "urlopen", fetch)
    monkeypatch.setattr(webui, "examples", lambda: {"examples": [{"id": "podcast"}]})
    payload = json.dumps(dict(id="podcast", url="https://attacker.test/audio")).encode()
    status, _, body = http(server, "/api/sample-reference", "POST", payload, headers)
    assert status == 400 and "hash mismatch" in json.loads(body)["error"]
    assert fetch.call_args.args[0] == "https://github.com/user-attachments/files/33393249/en0.wav"
    assert not studio.references


def test_immediate_exit_during_load_cleans_worker(studio, monkeypatch):
    process = fake_worker()
    process.poll.return_value = 2
    monkeypatch.setattr(web_service, "model_options", lambda data: dict(MODEL))
    monkeypatch.setattr(web_service.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="stopped"):
        studio.load({})
    assert studio.worker is None and studio.state["status"] == "unloaded"
    assert process.stdin.closed


def test_load_is_blocked_until_concurrent_unload_finishes(studio, monkeypatch):
    process = fake_worker()
    studio.worker = process
    entered, release = threading.Event(), threading.Event()

    def wait(timeout):
        entered.set()
        assert release.wait(5)
        return 0

    process.wait.side_effect = wait
    monkeypatch.setattr(web_service, "model_options", lambda data: dict(MODEL))
    spawn = Mock(return_value=fake_worker())
    monkeypatch.setattr(web_service.subprocess, "Popen", spawn)
    stopping = threading.Thread(target=studio.stop)
    stopping.start()
    try:
        assert entered.wait(2)
        assert studio.worker is None and studio.state["status"] == "unloading"
        with pytest.raises(ValueError, match="Unload"):
            studio.load({})
        spawn.assert_not_called()
    finally:
        release.set()
        stopping.join(3)
    assert studio.state["status"] == "unloaded"


def test_full_podcast_budget_is_explicit_and_unchunked():
    settings = web_config.generation_options(dict(
        text="A full podcast script.", max_frames=6144, chunk_length=0), False)
    assert settings["max_frames"] == 6144 and settings["chunk_length"] == 0
    assert web_config.generation_options(dict(text="Short default"), False)["max_frames"] == 512


def test_memory_telemetry_preserves_generation_status_and_message(studio):
    studio.state.update(status="generating", message="Generating speech…")
    memory = dict(gpu_free_bytes=1000, gpu_total_bytes=10000, headroom_percent=3)
    studio.handle_event(dict(memory=memory))
    state = studio.snapshot()
    assert state["status"] == "generating" and state["message"] == "Generating speech…"
    assert state["memory"] == memory


def test_reader_finds_telemetry_concatenated_with_native_progress(studio):
    memory = dict(gpu_free_bytes=1000, gpu_total_bytes=10000, headroom_percent=3)
    line = ' 12%|...| 711/6144 [00:12<01:30]DRAMA3_EVENT' + json.dumps(dict(memory=memory)) + '\n'
    process = fake_worker(line)
    studio.worker = process
    studio.state.update(status="generating", message="Generating speech…")
    studio.read_worker(process)
    assert studio.state["memory"] == memory
    assert studio.state["status"] == "error"
    assert "Generating speech…" in studio.state["message"]


def test_reader_extracts_native_frame_progress(studio):
    process = fake_worker(' 12%|...| 711/6144 [00:12<01:30, 12.34it/s]\n')
    studio.worker = process
    studio.state.update(status="generating", message="Generating speech…")
    studio.read_worker(process)
    assert studio.state["progress"] == dict(frames=711, budget=6144, frames_per_second=12.34)
    assert studio.state["phase"] == "generating"


def test_memory_event_preserves_phase_and_recoverable_error_is_visible(studio):
    studio.state.update(status="generating", phase="decoding", message="Decoding speech")
    studio.handle_event(dict(memory={"gpu_free_bytes": 1234}))
    assert studio.state["phase"] == "decoding" and studio.state["message"] == "Decoding speech"
    studio.handle_event(dict(status="ready", message="Generation failed", error="Frame limit reached"))
    assert studio.state["status"] == "ready"
    assert studio.state["error"] == studio.state["message"] == "Frame limit reached"


def saved_output(folder, identifier, settings, wav=True, mp3=False):
    directory = folder / identifier
    directory.mkdir(parents=True)
    (directory / "settings.json").write_text(
        settings if isinstance(settings, str) else json.dumps(settings), encoding="utf-8")
    if wav:
        (directory / "speech.wav").write_bytes(b"wave")
    if mp3:
        (directory / "speech.mp3").write_bytes(b"mp3")
    return directory


def test_restore_outputs_registers_downloads_and_latest_result(studio):
    import os

    first = saved_output(studio.output, "a" * 32, dict(metrics={"seed": 1}))
    latest = saved_output(studio.output, "b" * 32,
                          dict(metrics={"seed": 2}, actual_output_format="mp3"), mp3=True)
    os.utime(first / "settings.json", (1000, 1000))
    os.utime(latest / "settings.json", (2000, 2000))
    restarted = web_service.Studio(studio.output)
    result = restarted.snapshot()["result"]
    assert result["metrics"]["seed"] == 2
    assert result["audio_url"] == f"/outputs/{latest.name}/speech.mp3"
    assert result["wav_url"] == f"/outputs/{latest.name}/speech.wav"
    assert result["settings_url"] in restarted.exports
    assert f"/outputs/{first.name}/speech.wav" in restarted.exports
    assert restarted.snapshot()["status"] == "unloaded" and restarted.worker is None


@pytest.mark.parametrize("identifier,settings,wav", [
    ("not-a-uuid", {"metrics": {}}, True),
    ("c" * 32, {"metrics": {}}, False),
    ("d" * 32, "{invalid-json", True),
    ("e" * 32, {"seed": 123}, True),
    ("f" * 32, {"metrics": []}, True),
    ("2" * 32, "[]", True),
])
def test_restore_outputs_skips_incomplete_or_corrupt_records(studio, identifier, settings, wav):
    saved_output(studio.output, identifier, settings, wav=wav)
    restarted = web_service.Studio(studio.output)
    assert restarted.exports == {} and restarted.snapshot()["result"] is None


def test_restore_outputs_falls_back_to_wav_if_saved_mp3_is_missing(studio):
    directory = saved_output(studio.output, "1" * 32,
                             dict(metrics={"complete": True}, actual_output_format="mp3"))
    restarted = web_service.Studio(studio.output)
    result = restarted.snapshot()["result"]
    assert result["audio_url"] == result["wav_url"] == f"/outputs/{directory.name}/speech.wav"


def test_server_session_identifier_is_stable_and_unique_per_instance(studio):
    import uuid

    first = studio.snapshot()["server_session_id"]
    assert uuid.UUID(hex=first).hex == first
    assert studio.snapshot()["server_session_id"] == first
    restarted = web_service.Studio(studio.output)
    assert restarted.snapshot()["server_session_id"] != first


@pytest.mark.parametrize("asset", ["index.html", "app.js", "style.css", "podcast.json"])
def test_ui_revision_changes_when_asset_mtime_changes(studio, tmp_path, monkeypatch, asset):
    import os

    assets = tmp_path / "fake_module" / "web_assets"
    assets.mkdir(parents=True)
    for name in ("index.html", "app.js", "style.css", "podcast.json"):
        (assets / name).write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(web_service, "__file__", str(assets.parent / "web_service.py"))
    first = studio.snapshot()["ui_revision"]
    assert studio.snapshot()["ui_revision"] == first
    path = assets / asset
    stamp = path.stat().st_mtime_ns
    os.utime(path, ns=(stamp + 10**9, stamp + 10**9))
    assert studio.snapshot()["ui_revision"] != first
