"""Exercise stop conditions without allocating or stressing device memory."""
import subprocess
import sys
import json
import time
import shutil
from pathlib import Path

import pytest

from fish_drama3.watchdog import (
    Limits, guarded_job_lock, merge_child_telemetry, monitor,
    sample_memory, save_status, violation,
)


@pytest.mark.parametrize("terminate", [False, True])
def test_guarded_job_lock_excludes_process_and_releases_on_exit(
    tmp_path: Path, terminate: bool,
) -> None:
    lock = tmp_path / "shared-gpu.lock"
    code = (
        "from pathlib import Path; import sys; "
        "from fish_drama3.watchdog import guarded_job_lock; "
        "lease=guarded_job_lock(Path(sys.argv[1])); lease.__enter__(); "
        "print('LOCKED',flush=True); sys.stdin.readline(); lease.__exit__(None,None,None)"
    )
    owner = subprocess.Popen([sys.executable, "-c", code, str(lock)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert owner.stdout.readline().strip() == "LOCKED"
        borrower = subprocess.run([sys.executable, "-c", code, str(lock)],
                                  input="release\n", capture_output=True, text=True, timeout=5)
        assert borrower.returncode != 0 and "Another guarded GPU job" in borrower.stderr
        assert "No child was launched" in borrower.stderr
        if terminate:
            owner.terminate()
            owner.wait(timeout=5)
        else:
            owner.communicate("release\n", timeout=5)
            assert owner.returncode == 0
        with guarded_job_lock(lock):
            assert lock.is_file()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def test_busy_guarded_launch_does_not_sample_or_spawn(
    tmp_path: Path, monkeypatch,
) -> None:
    import fish_drama3.watchdog as watchdog
    lock = tmp_path / "shared-gpu.lock"
    monkeypatch.setattr(watchdog, "sample_memory",
                        lambda: pytest.fail("Sampled before acquiring lock"))
    monkeypatch.setattr(watchdog.subprocess, "Popen",
                        lambda *a, **kw: pytest.fail("Launched overlapping child"))
    with guarded_job_lock(lock):
        with pytest.raises(RuntimeError, match="Another guarded GPU job"):
            watchdog.launch_guarded([sys.executable, "-c", "pass"], tmp_path, Limits(), lock)


def test_status_publish_recovers_transient_windows_reader_lock(tmp_path: Path, monkeypatch) -> None:
    real = Path.replace
    attempts = []

    def transient(source: Path, destination: Path):
        attempts.append(source)
        if len(attempts) <= 2:
            raise PermissionError("Windows reader holds destination")
        return real(source, destination)

    monkeypatch.setattr(Path, "replace", transient)
    destination = tmp_path / "status.json"
    save_status(destination, dict(state="running"))
    assert json.loads(destination.read_text()) == dict(state="running")
    assert len(attempts) == 3


def test_actual_child_cuda_memory_takes_precedence(tmp_path: Path) -> None:
    path = tmp_path / "child_memory.json"
    path.write_text(json.dumps(dict(time=time.time(), gpu_free_bytes=2**29)))
    sample = dict(gpu_free_bytes=4 * 2**30, ram_free_bytes=80 * 2**30, commit_percent=20)
    merged = merge_child_telemetry(sample, path)
    assert merged["gpu_free_bytes"] == 2**29
    assert violation(merged, Limits()) == "gpu_headroom"
    path.write_text(json.dumps(dict(time=time.time() - 60, gpu_free_bytes=4 * 2**30)))
    with pytest.raises(RuntimeError, match="stale"):
        merge_child_telemetry(sample, path)


@pytest.mark.parametrize("field,value,reason", [
    ("gpu_free_bytes", 0, "gpu_headroom"),
    ("ram_free_bytes", 0, "ram_headroom"),
    ("commit_percent", 99, "system_commit"),
])
def test_memory_conditions(field: str, value: int, reason: str) -> None:
    sample = dict(gpu_free_bytes=8 * 2**30, ram_free_bytes=80 * 2**30, commit_percent=20)
    assert violation(sample, Limits()) is None
    sample[field] = value
    assert violation(sample, Limits()) == reason


@pytest.mark.parametrize("manual", [False, True])
def test_actual_child_is_stopped_without_memory_stress(tmp_path: Path, manual: bool) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    if manual:
        (tmp_path / "STOP").touch()
    unsafe = lambda: dict(gpu_free_bytes=0, ram_free_bytes=80 * 2**30, commit_percent=20)
    result = monitor(child, tmp_path, Limits(), reader=unsafe, interval=0.01)
    assert result["state"] == "stopped"
    assert result["reason"] == ("manual_stop" if manual else "gpu_headroom")
    assert child.poll() is not None
    assert (tmp_path / "watchdog.json").is_file()


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="Live NVIDIA telemetry unavailable")
def test_live_sensors_are_readable() -> None:
    sample = sample_memory()
    assert sample["gpu_free_bytes"] > 0
    assert sample["ram_free_bytes"] > 0
    assert 0 <= sample["commit_percent"] <= 100
