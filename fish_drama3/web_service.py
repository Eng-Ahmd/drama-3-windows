"""Serialized subprocess lifecycle for the local browser studio."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fish_drama3.web_config import generation_options, model_options


def find_encoder() -> str | None:
    encoder = shutil.which("ffmpeg")
    if encoder:
        try:
            result = subprocess.run([encoder, "-hide_banner", "-encoders"],
                                    capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0 and "libmp3lame" in result.stdout:
            return encoder
    return None


class Studio:
    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.encoder = find_encoder()
        self.lock = threading.RLock()
        self.worker: subprocess.Popen | None = None
        self.references: dict[str, Path] = {}
        self.exports: dict[str, Path] = {}
        self.pending: dict | None = None
        self.vitals_stop = threading.Event()
        self.vitals_thread: threading.Thread | None = None
        self.state = dict(status="unloaded", message="Load a model to begin.",
                          model=None, result=None, mp3_available=bool(self.encoder),
                          server_session_id=uuid.uuid4().hex)
        self.restore_outputs()

    def restore_outputs(self) -> None:
        """Keep completed local downloads accessible after a studio restart."""
        latest = 0.0
        for directory in self.output.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            if not directory.resolve().is_relative_to(self.output):
                continue
            settings = directory / "settings.json"
            wav = directory / "speech.wav"
            if not settings.is_file() or not wav.is_file() or settings.stat().st_size > 4 * 2**20:
                continue
            try:
                saved = json.loads(settings.read_text(encoding="utf-8"))
                if not isinstance(saved, dict):
                    continue
                metrics = saved["metrics"]
                if not isinstance(metrics, dict):
                    continue
            except (OSError, ValueError, KeyError):
                continue
            audio = directory / "speech.mp3" if saved.get("actual_output_format") == "mp3" else wav
            if not audio.is_file():
                audio = wav
            paths = dict(audio_url=audio, wav_url=wav, settings_url=settings)
            if any(path.is_symlink() for path in paths.values()):
                continue
            urls = {key: f"/outputs/{directory.name}/{path.name}" for key, path in paths.items()}
            self.exports.update({urls[key]: path for key, path in paths.items()})
            if settings.stat().st_mtime > latest:
                latest = settings.stat().st_mtime
                self.state["result"] = dict(urls, metrics=metrics)

    def start_vitals(self) -> None:
        from fish_drama3.web_vitals import sample_vitals

        def monitor() -> None:
            while not self.vitals_stop.is_set():
                try:
                    sample = sample_vitals()
                except Exception as error:
                    sample = dict(error=str(error), updated_at=time.time())
                with self.lock:
                    self.state["vitals"] = sample
                self.vitals_stop.wait(2)

        self.vitals_thread = threading.Thread(target=monitor, name="studio-vitals", daemon=True)
        self.vitals_thread.start()

    def shutdown(self) -> None:
        self.vitals_stop.set()
        self.stop()
        if self.vitals_thread:
            self.vitals_thread.join(timeout=5)

    def snapshot(self) -> dict:
        with self.lock:
            result = json.loads(json.dumps(self.state))
        assets = Path(__file__).parent / "web_assets"
        stamps = [str((assets / name).stat().st_mtime_ns) for name in
                  ("index.html", "app.js", "style.css", "podcast.json") if (assets / name).is_file()]
        result["ui_revision"] = hashlib.sha256("|".join(stamps).encode()).hexdigest()[:16]
        return result

    def send(self, payload: dict) -> None:
        if not self.worker or self.worker.poll() is not None:
            raise RuntimeError("Model worker has stopped. Load the model again.")
        self.worker.stdin.write(json.dumps(payload) + "\n")
        self.worker.stdin.flush()

    def load(self, data: dict) -> dict:
        options = model_options(data)
        with self.lock:
            if self.worker is not None or self.state["status"] not in ("unloaded", "error"):
                raise ValueError("Unload the current model before changing model settings")
            self.state.update(status="loading", message="Verifying files and loading model…",
                              model=options, phase="verifying", error=None, started_at=time.time())
            try:
                self.worker = subprocess.Popen(
                    [sys.executable, "-u", "-m", "fish_drama3.web_worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.send(dict(action="load", **options))
            except (OSError, RuntimeError):
                self.stop("Could not start model worker")
                raise
            threading.Thread(target=self.read_worker, args=(self.worker,), daemon=True).start()
        return self.snapshot()

    def generate(self, data: dict) -> dict:
        options = generation_options(data, bool(self.encoder))
        from fish_drama3.web_planning import generation_plan
        plan = generation_plan(options)
        if plan["risk"] == "high" and data.get("plan_acknowledged") is not True:
            raise ValueError("Frame budget risk: review the preflight estimate before generating. "
                             "Use smart segments and the recommended total budget, or acknowledge "
                             "the risk to run with these settings.")
        reference_id = data.get("reference_id")
        if reference_id is not None and not isinstance(reference_id, str):
            raise ValueError("Invalid reference identifier")
        reference = self.references.get(reference_id)
        if reference_id and reference is None:
            raise ValueError("Reference expired; upload the reference audio again")
        if bool(reference) != bool(options["reference_text"]):
            raise ValueError("Supply reference audio and its transcript together")
        with self.lock:
            if self.state["status"] != "ready":
                raise ValueError("Load a model and wait until it is ready")
            identifier = uuid.uuid4().hex
            directory = self.output / identifier
            directory.mkdir()
            self.pending = dict(options, model={key: self.state["model"][key] for key in
                                              ("bundle", "codec", "threads", "gpu_expert_layers",
                                               "placement_mode", "headroom_percent")
                                              if key in self.state["model"]},
                                reference_id=reference_id, id=identifier)
            try:
                self.send(dict(action="generate", **options, output=str(directory / "speech.wav"),
                               reference_audio=str(reference) if reference else None))
            except (RuntimeError, OSError):
                self.pending = None
                self.state.update(status="error", message="Worker stopped. Unload and reload the model.")
                raise
            self.state.update(status="generating", message="Generating speech…", result=None,
                              started_at=time.time(), progress=None, phase="encoding_reference",
                              error=None, plan=plan)
        return self.snapshot()

    def plan(self, data: dict) -> dict:
        from fish_drama3.web_planning import generation_plan
        return generation_plan(generation_options(data, bool(self.encoder)))

    def read_worker(self, worker: subprocess.Popen) -> None:
        log_path = self.output / "worker.log"
        with log_path.open("a", encoding="utf-8") as log:
            for line in worker.stdout:
                log.write(line)
                log.flush()
                marker = line.find("DRAMA3_EVENT")
                if marker < 0:
                    match = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
                    if match:
                        with self.lock:
                            if self.worker is worker and self.state["status"] == "generating":
                                speed = re.search(r"([\d.]+)it/s", line)
                                self.state["phase"] = "generating"
                                self.state["message"] = "Generating speech…"
                                progress = dict(self.state.get("progress") or {})
                                completed = progress.get("completed_frames", 0)
                                progress.update(frames=completed + int(match[1]),
                                    segment_frames=int(match[1]),
                                    budget=self.pending["max_frames"] if self.pending else int(match[2]),
                                    frames_per_second=float(speed[1]) if speed else None)
                                self.state["progress"] = progress
                    continue
                try:
                    event = json.loads(line[marker + len("DRAMA3_EVENT"):].strip())
                    with self.lock:
                        if self.worker is not worker:
                            continue
                        pending = dict(self.pending) if self.pending else None
                        if event.get("result"):
                            self.state.update(phase="exporting", message="Encoding output and saving settings…")
                    exported = self.prepare_result(event["result"], pending) if event.get("result") else None
                    with self.lock:
                        if self.worker is not worker:
                            continue
                        if exported:
                            self.state["result"] = exported
                            self.pending = None
                        self.handle_event(event)
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    with self.lock:
                        if self.worker is worker:
                            self.state.update(status="ready", message=f"Export failed: {error}")
            code = worker.wait()
        with self.lock:
            if self.worker is worker:
                self.worker = None
                self.state.update(status="error", model=None,
                                  message=f"Model worker exited ({code}). {self.state['message']}")

    def handle_event(self, event: dict) -> None:
        self.state["status"] = event.get("status", self.state["status"])
        if "error" in event or "message" in event:
            self.state["message"] = event.get("error") or event.get("message", "")
        if event.get("memory"):
            self.state["memory"] = event["memory"]
        if event.get("performance"):
            self.state["performance"] = event["performance"]
        if event.get("phase"):
            self.state["phase"] = event["phase"]
        if event.get("progress"):
            self.state["progress"] = dict(self.state.get("progress") or {}, **event["progress"])
        if "error" in event:
            self.state["error"] = event["error"]
            self.state["phase"] = "error"
        if event.get("result"):
            self.state.update(phase="complete", error=None)
        if event.get("model"):
            self.state["model"] = event["model"]

    def prepare_result(self, result: dict, pending: dict) -> dict:
        settings = dict(pending)
        identifier = settings.pop("id")
        wav = self.output / identifier / "speech.wav"
        if Path(result["output"]).resolve() != wav or not wav.is_file():
            raise ValueError("Worker returned an unexpected output path")
        audio = wav
        warning = None
        if settings["output_format"] == "mp3":
            audio, warning = self.encode_mp3(wav, settings["mp3_bitrate"])
        settings.update(schema="fish-drama3-studio-v1", metrics=result["metrics"],
                        reference_note="Upload the same reference audio when restoring settings.",
                        actual_output_format=audio.suffix[1:])
        settings_path = wav.with_name("settings.json")
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        urls = {"audio_url": audio, "wav_url": wav, "settings_url": settings_path}
        with self.lock:
            for path in (*urls.values(), wav.with_suffix(".json")):
                self.exports[f"/outputs/{identifier}/{path.name}"] = path
        exported = {key: f"/outputs/{identifier}/{path.name}" for key, path in urls.items()}
        exported.update(metrics=result["metrics"], warning=warning)
        return exported

    def encode_mp3(self, wav: Path, bitrate: int) -> tuple[Path, str | None]:
        mp3 = wav.with_suffix(".mp3")
        try:
            subprocess.run([self.encoder, "-nostdin", "-v", "error", "-n", "-i", str(wav),
                            "-codec:a", "libmp3lame", "-b:a", f"{bitrate}k", str(mp3)],
                           capture_output=True, timeout=120, check=True)
            return mp3, None
        except (OSError, subprocess.SubprocessError):
            mp3.unlink(missing_ok=True)
            return wav, "MP3 encoding failed; the original WAV is available."

    def stop(self, message: str = "Model unloaded. VRAM released.") -> dict:
        with self.lock:
            if self.state["status"] == "unloading":
                return self.snapshot()
            worker, self.worker = self.worker, None
            self.state.update(status="unloading", message="Releasing model memory…")
        if worker is not None:
            if worker.poll() is None:
                worker.terminate()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10)
            worker.stdin.close()
        with self.lock:
            self.pending = None
            self.state.update(status="unloaded", message=message, model=None, memory=None,
                              phase="idle", error=None)
        return self.snapshot()
