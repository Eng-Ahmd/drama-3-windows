"""Read real host and NVIDIA statistics for the local studio dashboard."""
from __future__ import annotations

import subprocess
import time

import psutil


def sample_vitals() -> dict:
    memory = psutil.virtual_memory()
    result = dict(cpu_percent=psutil.cpu_percent(interval=.1),
                  cpu_threads=psutil.cpu_count(), ram_percent=memory.percent,
                  ram_used_bytes=memory.used, ram_total_bytes=memory.total,
                  updated_at=time.time())
    fields = "utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw"
    try:
        process = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                                  "--format=csv,noheader,nounits", "--id=0"],
                                 capture_output=True, text=True, timeout=3, check=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        values = [value.strip() for value in process.stdout.strip().splitlines()[0].split(",")]
        keys = ("gpu_utilization", "gpu_used_bytes", "gpu_total_bytes", "gpu_free_bytes",
                "gpu_temperature", "gpu_power_watts")
        for index, (key, value) in enumerate(zip(keys, values, strict=True)):
            try:
                number = float(value)
                result[key] = number * 2**20 if index in (1, 2, 3) else number
            except ValueError:
                result[key] = None
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as error:
        result["error"] = f"GPU statistics unavailable: {error}"
    return result
