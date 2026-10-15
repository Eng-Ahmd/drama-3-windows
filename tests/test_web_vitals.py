"""Host/GPU dashboard statistics use actual sensor values and tolerate absence."""
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest

from fish_drama3 import web_vitals


@pytest.fixture(autouse=True)
def host_sensors(monkeypatch):
    monkeypatch.setattr(web_vitals.psutil, "virtual_memory", lambda: SimpleNamespace(
        percent=42.5, used=7 * 2**30, total=16 * 2**30))
    monkeypatch.setattr(web_vitals.psutil, "cpu_percent", lambda interval: 23.5)
    monkeypatch.setattr(web_vitals.psutil, "cpu_count", lambda: 16)
    monkeypatch.setattr(web_vitals.time, "time", lambda: 123456.0)


def test_csv_gpu_sensors_convert_mib_to_bytes(monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout=" 87, 12000, 16384, 4384, 65, 123.45 \n"))
    monkeypatch.setattr(web_vitals.subprocess, "run", run)
    result = web_vitals.sample_vitals()
    assert result["gpu_utilization"] == 87
    assert result["gpu_used_bytes"] == 12000 * 2**20
    assert result["gpu_total_bytes"] == 16384 * 2**20
    assert result["gpu_free_bytes"] == 4384 * 2**20
    assert result["gpu_temperature"] == 65 and result["gpu_power_watts"] == 123.45
    assert result["cpu_percent"] == 23.5 and result["cpu_threads"] == 16
    assert result["ram_used_bytes"] == 7 * 2**30 and result["ram_percent"] == 42.5
    assert result["updated_at"] == 123456.0 and "error" not in result
    assert "--id=0" in run.call_args.args[0]
    assert run.call_args.kwargs["timeout"] == 3


def test_unsupported_gpu_sensors_are_none(monkeypatch):
    monkeypatch.setattr(web_vitals.subprocess, "run", Mock(return_value=SimpleNamespace(
        stdout="N/A, 1024, 16384, 15360, [Not Supported], N/A\n")))
    result = web_vitals.sample_vitals()
    assert result["gpu_utilization"] is None and result["gpu_temperature"] is None
    assert result["gpu_power_watts"] is None and result["gpu_used_bytes"] == 1024 * 2**20
    assert "error" not in result


@pytest.mark.parametrize("error", [
    FileNotFoundError("nvidia-smi unavailable"), subprocess.TimeoutExpired("nvidia-smi", 3),
])
def test_missing_gpu_preserves_host_statistics(monkeypatch, error):
    monkeypatch.setattr(web_vitals.subprocess, "run", Mock(side_effect=error))
    result = web_vitals.sample_vitals()
    assert "GPU statistics unavailable" in result["error"]
    assert result["cpu_percent"] == 23.5 and result["ram_total_bytes"] == 16 * 2**30
    assert "gpu_utilization" not in result


@pytest.mark.parametrize("output", ["", "1,2,3\n"])
def test_malformed_gpu_csv_reports_error(monkeypatch, output):
    monkeypatch.setattr(web_vitals.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=output)))
    assert "error" in web_vitals.sample_vitals()
