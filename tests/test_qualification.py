"""Qualification logic tests; these do not claim model qualification."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fish_drama3 import qualification as qa


def test_frozen_preview_has_required_coverage() -> None:
    cases = qa.frozen_cases(qa.CORPUS)
    assert len(cases) == 12
    assert {case.get("language") for case in cases} >= {"en", "zh"}
    assert sum(case.get("reference") == "en0" for case in cases) == 2
    assert sum(bool(case.get("expect_reject")) for case in cases) == 3
    assert qa.scheduled_case(cases, 12, "preview")["id"] == "en-plain"
    for index in range(16, 100, 6):
        assert qa.scheduled_case(cases, index, "soak")["expect_reject"]
        assert qa.scheduled_case(cases, index + 1, "soak")["id"] == "en-plain"


def test_checkpoint_truncation_and_corruption(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    qa.append_event(path, {"index": 0, "event": "completed"})
    with path.open("ab") as handle:
        handle.write(b'{"index":1')
    assert qa.read_events(path) == [dict(index=0, event="completed")]
    assert path.read_bytes().endswith(b"\n")
    with path.open("ab") as handle:
        handle.write(b"corrupt\n")
    with pytest.raises(ValueError, match="Corrupt completed"):
        qa.read_events(path)


def test_drift_uses_private_and_actual_device_allocations() -> None:
    baseline = dict(cache_capacity=512, private_bytes=1000,
                    torch_allocated_bytes=1000, cuda_used_bytes=1000)
    current = {**baseline, "private_bytes": 1256, "cuda_used_bytes": 1200}
    assert qa.memory_drift(baseline, current, 256)["passed"]
    current["cuda_used_bytes"] = 1257
    assert not qa.memory_drift(baseline, current, 256)["passed"]
    current["cache_capacity"] = 768
    assert qa.memory_drift(baseline, current, 256)["status"] == "declared_cache_growth"
    missing = {**baseline, "private_bytes": None}
    assert not qa.memory_drift(missing, baseline, 256)["passed"]


def test_wer_frozen_normalization_and_edit_distance() -> None:
    assert qa.word_error_rate("Hello, WORLD!", "hello world") == 0
    assert qa.word_error_rate("one two three", "one four") == pytest.approx(2 / 3)
    assert qa.word_error_rate("one", "one two three") == 2
    with pytest.raises(ValueError):
        qa.word_error_rate("", "anything")


def test_watchdog_thresholds_come_from_parent(monkeypatch) -> None:
    monkeypatch.setenv("FISH_DRAMA3_WATCHDOG_GPU_GIB", "1.5")
    assert qa.watchdog_thresholds()["gpu_free_gib"] == 1.5
    monkeypatch.setenv("FISH_DRAMA3_WATCHDOG_GPU_GIB", "nan")
    with pytest.raises(ValueError):
        qa.watchdog_thresholds()


def test_resume_requires_identical_identity(tmp_path, monkeypatch) -> None:
    options = SimpleNamespace(output=tmp_path, resume=False, bundle=tmp_path / "bundle",
                              codec=tmp_path / "codec.gguf")
    monkeypatch.setattr(qa, "verify_assets", lambda *args: {"actual_model_sha256": "verified"})
    monkeypatch.setattr(qa, "source_identity", lambda _: {"frozen": 1})
    manifest, events = qa.initialize_run(options)
    assert not events and not manifest["qualified"]
    options.resume = True
    manifest, _ = qa.initialize_run(options)
    assert len(manifest["segments"]) == 2
    monkeypatch.setattr(qa, "source_identity", lambda _: {"frozen": 2})
    with pytest.raises(ValueError, match="identical frozen"):
        qa.initialize_run(options)


def test_failed_completed_preview_cannot_become_green_on_resume(tmp_path, monkeypatch) -> None:
    cases = [dict(id="short", text="Hi", max_frames=2, seed=1)]
    event = dict(index=1, event="completed", case_id="short", seed=1, passed=False)
    options = SimpleNamespace(output=tmp_path, mode="preview", resume=True)
    monkeypatch.setattr(qa, "record_request", lambda *args: pytest.fail("Failed run was executed"))
    result = qa.run_requests(options, cases, None, None, [event])
    assert result["status"] == "failed" and not result["preview_checks_passed"]
    assert result["completed_requests"] == 0 and result["negative_recovery_cycles"] == 0
    qa.append_event(tmp_path / "requests.jsonl", event)
    with pytest.raises(ValueError, match="failed run cannot be resumed"):
        qa.initialize_run(options)


def test_actual_asset_verification_is_mandatory_before_identity(tmp_path, monkeypatch) -> None:
    options = SimpleNamespace(output=tmp_path, resume=False, bundle=tmp_path / "bundle",
                              codec=tmp_path / "codec.gguf")
    def invalid_assets(*args):
        raise ValueError("actual payload hash mismatch")
    monkeypatch.setattr(qa, "verify_assets", invalid_assets)
    monkeypatch.setattr(qa, "source_identity", lambda _: pytest.fail("Identity preceded payload check"))
    with pytest.raises(ValueError, match="actual payload hash"):
        qa.initialize_run(options)


def test_length_finish_is_failure(monkeypatch) -> None:
    from fish_drama3 import hybrid

    rows = torch.tensor([[248100, 1] + [0] * 9], dtype=torch.int64)
    monkeypatch.setattr(hybrid, "generate_drama3", lambda *args, **kwargs: (rows, "length"))
    monkeypatch.setattr(qa, "case_prompt", lambda *args: torch.zeros((2, 11), dtype=torch.int64))
    model = SimpleNamespace(config=SimpleNamespace(text=SimpleNamespace(max_position_embeddings=8192)))
    codec = SimpleNamespace(decode=lambda _: np.ones(2048, dtype=np.float32))
    record, _, _ = qa.execute_case(dict(seed=1, max_frames=128), model, codec)
    assert not record["passed"] and record["finish_reason"] == "length"


def test_soak_reports_unavailable_gates_without_qualifying(tmp_path, monkeypatch) -> None:
    cases = [dict(id="short", text="Hi", max_frames=2, seed=1)]
    options = SimpleNamespace(output=tmp_path, mode="preview")
    memory = dict(cache_capacity=2, private_bytes=10,
                  torch_allocated_bytes=10, cuda_used_bytes=10)
    monkeypatch.setattr(qa, "record_request", lambda index, *args:
        dict(index=index, case_id="short", seed=1, event="completed", passed=True,
             memory=memory))
    result = qa.run_requests(options, cases, None, None, [])
    assert result["preview_checks_passed"] and result["qualified"] is False
    assert "unavailable" in result["cancellation_recovery"]
    assert "48 GB" in result["production_reference"]
