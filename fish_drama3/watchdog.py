"""Run qualification in a bounded child process, with a manual STOP switch."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                *[(name, ctypes.c_ulonglong) for name in
                  ("total_ram", "free_ram", "total_commit", "free_commit",
                   "total_virtual", "free_virtual", "free_extended")]]


@dataclass(frozen=True)
class Limits:
    gpu_free_bytes: int = 2**30
    ram_free_bytes: int = 8 * 2**30
    commit_percent: float = 90.0


def sample_memory() -> dict:
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot read Windows memory status")
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=True)
    values = [int(line.strip()) * 2**20 for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError("Watchdog requires exactly one visible NVIDIA GPU")
    return dict(gpu_free_bytes=values[0], ram_free_bytes=status.free_ram,
                commit_percent=100 * (1 - status.free_commit / max(status.total_commit, 1)),
                time=time.time())


def violation(sample: dict, limits: Limits) -> str | None:
    if sample["gpu_free_bytes"] < limits.gpu_free_bytes:
        return "gpu_headroom"
    if sample["ram_free_bytes"] < limits.ram_free_bytes:
        return "ram_headroom"
    if sample["commit_percent"] > limits.commit_percent:
        return "system_commit"
    return None


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                   capture_output=True, timeout=10, check=False)
    if child.poll() is None:
        child.kill()
    child.wait(timeout=10)


def save_status(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)  # Windows readers may briefly hold a non-delete-shared handle.


def start_child_telemetry() -> tuple[threading.Event, threading.Thread]:
    """Report the child's actual CUDA budget; WDDM NVML alone can overstate it."""
    location = os.environ.get("FISH_DRAMA3_WATCHDOG_TELEMETRY")
    if not location:
        raise RuntimeError("Missing external watchdog telemetry path")
    output = Path(location)
    stop = threading.Event()

    def report() -> None:
        import torch
        while not stop.is_set():
            record = dict(time=time.time())
            try:
                free, total = torch.cuda.mem_get_info(0)
                record.update(gpu_free_bytes=free, gpu_total_bytes=total)
            except Exception as error:
                record["error"] = str(error)
            save_status(output, record)
            stop.wait(0.5)

    thread = threading.Thread(target=report, name="CUDA-memory-telemetry", daemon=True)
    thread.start()
    return stop, thread


def merge_child_telemetry(sample: dict, path: Path) -> dict:
    if not path.exists():
        return sample
    telemetry = json.loads(path.read_text(encoding="utf-8"))
    if "error" in telemetry or time.time() - telemetry["time"] > 10:
        raise RuntimeError("Child CUDA telemetry failed or became stale")
    sample["nvml_gpu_free_bytes"] = sample["gpu_free_bytes"]
    sample["child_gpu_free_bytes"] = telemetry["gpu_free_bytes"]
    sample["gpu_free_bytes"] = min(sample["gpu_free_bytes"], telemetry["gpu_free_bytes"])
    return sample


def monitor(child: subprocess.Popen, output: Path, limits: Limits,
            reader: Callable[[], dict] = sample_memory, interval: float = 1.0) -> dict:
    record = dict(pid=child.pid, state="running", limits=limits.__dict__)
    launched = time.monotonic()
    try:
        while child.poll() is None:
            if (output / "STOP").exists():
                record.update(state="stopped", reason="manual_stop")
                break
            try:
                if not (output / "child_memory.json").exists() and time.monotonic() - launched > 30:
                    raise RuntimeError("Child CUDA telemetry handshake missing after 30 seconds")
                sample = merge_child_telemetry(reader(), output / "child_memory.json")
                record["last_sample"] = sample
                reason = violation(sample, limits)
            except Exception as error:
                reason = "sensor_failure"
                record["sensor_error"] = str(error)
            if reason:
                record.update(state="stopped", reason=reason)
                break
            save_status(output / "watchdog.json", record)
            time.sleep(interval)
        else:
            record.update(state="finished", exit_code=child.returncode)
    except KeyboardInterrupt:
        record.update(state="stopped", reason="keyboard_interrupt")
    finally:
        if child.poll() is None:
            stop_child(child)
        record["exit_code"] = child.returncode
        save_status(output / "watchdog.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-free-gib", type=float, default=1)
    parser.add_argument("--ram-free-gib", type=float, default=8)
    parser.add_argument("--commit-percent", type=float, default=90)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.gpu_free_gib <= 0 or args.ram_free_gib <= 0:
        parser.error("A command and positive memory reserves are required")
    if not 0 < args.commit_percent < 100:
        parser.error("Commit threshold must be between 0 and 100")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "STOP").exists():
        parser.error("Manual STOP switch already exists")
    limits = Limits(int(args.gpu_free_gib * 2**30), int(args.ram_free_gib * 2**30),
                    args.commit_percent)
    before = sample_memory()
    if violation(before, limits):
        raise SystemExit("Watchdog preflight rejected available memory")
    (args.output / "child_memory.json").unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(FISH_DRAMA3_WATCHDOG_ACTIVE="1",
                       FISH_DRAMA3_WATCHDOG_TELEMETRY=str((args.output / "child_memory.json").resolve()),
                       FISH_DRAMA3_WATCHDOG_GPU_GIB=str(args.gpu_free_gib),
                       FISH_DRAMA3_WATCHDOG_RAM_GIB=str(args.ram_free_gib),
                       FISH_DRAMA3_WATCHDOG_COMMIT_PERCENT=str(args.commit_percent))
    with (args.output / "child.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = monitor(child, args.output, limits)
    print(json.dumps(record), flush=True)
    raise SystemExit(record["exit_code"] if record["state"] == "finished" else 125)


if __name__ == "__main__":
    main()
