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
    from fish_drama3 import generation

    rows = torch.tensor([[248100, 1] + [0] * 9], dtype=torch.int64)
    segments = [dict(finish_reason="length", frames=1, prompt_rows=2)]
    monkeypatch.setattr(generation, "generate_text", lambda *args: (rows, "length", segments))
    monkeypatch.setattr(qa, "case_prompt", lambda *args: torch.zeros((2, 11), dtype=torch.int64))
    model = SimpleNamespace(config=SimpleNamespace(text=SimpleNamespace(max_position_embeddings=8192)))
    codec = SimpleNamespace(decode=lambda _: np.ones(2048, dtype=np.float32))
    record, _, _ = qa.execute_case(dict(seed=1, text="Hello.", max_frames=128), model, codec)
    assert not record["passed"] and record["finish_reason"] == "length"
    assert record["text_segments"] == segments and not record["complete"]


def test_public_chunk_history_and_total_budget_are_preserved(monkeypatch) -> None:
    from fish_drama3 import generation

    prompts, budgets = [], []
    references = [torch.ones((10, 3), dtype=torch.int64)]
    monkeypatch.setattr(qa, "case_references", lambda _: (["Pinned reference"], references))
    def build_prompt(text, tokenizer, config, texts, codes, history):
        prompts.append(dict(text=text, texts=texts, codes=codes, history=history))
        return torch.zeros((2, 11), dtype=torch.int64)
    def generate(model, prompt, max_new_tokens, **kwargs):
        budgets.append(max_new_tokens)
        rows = torch.tensor([[248100, 1] + [0] * 9, [248101, 2] + [0] * 9])
        return rows, "eos"
    monkeypatch.setattr(generation, "build_drama3_prompt", build_prompt)
    monkeypatch.setattr(generation, "generate_drama3", generate)
    model = SimpleNamespace(tokenizer=None, config=None)
    text = "<|speaker:1|> First sentence has words. Second sentence has words. Third sentence has words."
    frames, reason, segments, complete = qa.case_generation(dict(text=text, max_frames=4), model, 48)
    assert reason == "length" and not complete and len(frames) == 4
    assert budgets == [4, 2] and len(segments) == 2
    assert prompts[0]["history"] == [] and len(prompts[1]["history"]) == 1
    assert prompts[1]["text"].startswith("<|speaker:1|>")
    assert prompts[1]["texts"] == ["Pinned reference"] and prompts[1]["codes"] is references
    history_text, history_codes = prompts[1]["history"][0]
    assert history_text == prompts[0]["text"] and history_codes.shape == (10, 2)
    torch.testing.assert_close(history_codes[0], torch.tensor([1, 2]))


def test_false_eos_cannot_hide_missing_text_segments(monkeypatch) -> None:
    from fish_drama3 import generation

    rows = torch.tensor([[248100, 1] + [0] * 9])
    segments = [dict(finish_reason="eos", frames=1)]
    monkeypatch.setattr(generation, "generate_text", lambda *args: (rows, "eos", segments))
    text = "First sentence has words. Second sentence has words. Third sentence has words."
    _, reason, _, complete = qa.case_generation(dict(text=text, max_frames=100), None, 32)
    assert reason == "eos" and not complete


def test_legacy_control_uses_same_public_helper_and_one_segment(monkeypatch) -> None:
    from fish_drama3 import generation

    rows = torch.tensor([[248100, 1] + [0] * 9])
    def generate(model, args, texts, codes):
        assert args.chunk_length == 0 and args.max_frames == 10
        return rows, "eos", [dict(finish_reason="eos", frames=1)]
    monkeypatch.setattr(generation, "generate_text", generate)
    _, _, _, complete = qa.case_generation(dict(text="Hello.", max_frames=10), None, 0)
    assert complete


def test_expected_negative_is_rejected_before_public_generation(monkeypatch) -> None:
    def rejected_prompt(*args):
        raise ValueError("Invalid canonical speaker")
    monkeypatch.setattr(qa, "case_prompt", rejected_prompt)
    monkeypatch.setattr(qa, "case_generation", lambda *args: pytest.fail("Negative request reached generation"))
    record, rows, audio = qa.execute_case(dict(seed=1, max_frames=128, expect_reject=True), None, None)
    assert record["passed"] and record["rejected"] and rows is None and audio is None


@pytest.mark.parametrize("length", [0, 32, 200, 2000])
def test_chunk_length_cli_accepts_public_bounds(length, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["qualification", "--output", str(tmp_path),
                                     "--chunk-length", str(length)])
    assert qa.parse_options().chunk_length == length


@pytest.mark.parametrize("length", [-1, 1, 31, 2001])
def test_chunk_length_cli_rejects_invalid_bounds(length, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["qualification", "--output", str(tmp_path),
                                     "--chunk-length", str(length)])
    with pytest.raises(SystemExit) as error:
        qa.parse_options()
    assert error.value.code == 2


def test_chunking_identity_records_schema_source_hashes_and_length(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    codec_folder = tmp_path / "codec"
    bundle.mkdir()
    codec_folder.mkdir()
    (bundle / "manifest.json").write_text(json.dumps(dict(model_sha256="model")))
    (codec_folder / "manifest.json").write_text(json.dumps(dict(sha256="codec")))
    for relative in ("fish_drama3/hybrid.py", "fish_drama3/generation.py",
                     "fish_drama3/text_chunks.py", "models/comparison/en0-codes.npy", "cases.jsonl"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    monkeypatch.setattr(qa, "ROOT", tmp_path)
    options = SimpleNamespace(bundle=bundle, codec=codec_folder / "codec.gguf",
        corpus=tmp_path / "cases.jsonl", mode="preview", gpu_expert_layers=12,
        threads=8, soak_seconds=7200, min_requests=200, failure_cycles=50, chunk_length=200)
    initial = qa.source_identity(options)
    assert initial["schema"] == "fish-drama3-qualification-v2-chunked"
    assert initial["chunk_length"] == 200
    assert initial["generation_source_sha256"] == qa.sha_file(tmp_path / "fish_drama3/generation.py")
    assert initial["text_chunks_source_sha256"] == qa.sha_file(tmp_path / "fish_drama3/text_chunks.py")
    options.chunk_length = 0
    assert qa.source_identity(options) != initial
    options.chunk_length = 200
    (tmp_path / "fish_drama3/generation.py").write_bytes(b"changed")
    assert qa.source_identity(options) != initial


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
