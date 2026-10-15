"""Parity checks must reject mismatched benchmark evidence."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_encoder_comparison_requires_identical_codes(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts/benchmark-codec.py"
    spec = importlib.util.spec_from_file_location("codec_benchmark", source)
    assert spec and spec.loader
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    reports = [tmp_path / "baseline.json", tmp_path / "candidate.json"]
    for path in reports:
        path.write_text(json.dumps(dict(cases=[], seam=None)), encoding="utf-8")
        np.save(path.with_name(f"{path.stem}-encoded.npy"), np.zeros((10, 2), np.int32))
    args = argparse.Namespace(check_encode=True, atol=1e-6, rtol=1e-5)
    assert benchmark.compare_reports(reports, args)["passed"]
    np.save(tmp_path / "candidate-encoded.npy", np.ones((10, 2), np.int32))
    result = benchmark.compare_reports(reports, args)
    assert not result["encoder_codes_identical"]
    assert not result["passed"]


def test_window_study_preserves_baseline_when_candidate_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts/benchmark-codec.py"
    spec = importlib.util.spec_from_file_location("codec_window_benchmark", source)
    assert spec and spec.loader
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    rows = np.zeros((2049, 11), np.int32)
    rows[:, 0] = 248099
    fixture = tmp_path / "codes.npy"
    np.save(fixture, rows)
    monkeypatch.setattr(benchmark, "CodecBridge", lambda *_: nullcontext(None))

    def decode(_bridge, _rows, window, _threads):
        if window != 1792:
            raise RuntimeError("candidate interrupted")
        return np.zeros(64, np.float32), 12.0

    monkeypatch.setattr(benchmark, "window_decode", decode)
    output = tmp_path / "study.json"
    args = argparse.Namespace(codes=fixture, study_frames=2049, threads=[8],
                              output=output, codec=Path("codec"),
                              baseline=Path("baseline"), window_sizes=[256])
    with pytest.raises(RuntimeError, match="candidate interrupted"):
        benchmark.window_study(args)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["partial"]
    assert report["baseline_seconds"] == 12.0
    assert output.with_suffix(".baseline.npy").exists()


def test_single_decode_measurement_uses_first_timing(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts/benchmark-codec.py"
    spec = importlib.util.spec_from_file_location("codec_single_benchmark", source)
    assert spec and spec.loader
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class Bridge:
        calls = 0

        def decode(self, rows, threads):
            self.calls += 1
            return np.ones(len(rows) * 2048, np.float32)

    bridge = Bridge()
    args = argparse.Namespace(repeats=0, output=tmp_path / "single.json")
    result = benchmark.case(bridge, np.zeros((2, 11), np.int32), 8, args)
    assert bridge.calls == 1
    assert result["repeat_seconds"] == []
    assert result["timing_basis"] == "first"
    assert result["median_repeat_seconds"] == result["first_seconds"]
