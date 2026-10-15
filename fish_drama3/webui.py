"""Local browser interface for Drama 3: python -m fish_drama3.webui."""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fish_drama3.web_service import Studio

ASSETS = Path(__file__).parent / "web_assets"
ROOT = Path(__file__).resolve().parents[1]


def examples() -> dict:
    short = dict(id="welcome", title="A first voice", text="Hello. Every voice has a story to tell.",
                 description="A short first generation.", settings={})
    path = ASSETS / "podcast.json"
    if not path.is_file():
        path = ROOT / "examples/samples/podcast.json"
    records = [short]
    if path.is_file():
        sample = json.loads(path.read_text(encoding="utf-8"))
        sample["settings"] = {key: sample[key] for key in
                              ("seed", "temperature", "top_p", "top_k")}
        sample["settings"].update(max_frames=6144, chunk_length=0, generation_mode="single")
        sample["description"] = ("592-word podcast · single-pass generation · seed 123. "
                                 "Evaluate each output before sharing.")
        sample["audio_url"] = sample.get("oneshot_preview_url", "")
        sample["transcript_url"] = sample["preview_url"].replace(".mp3", ".txt")
        records.append(sample)
    return dict(examples=records)


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], studio: Studio) -> None:
        self.studio = studio
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, message_format: str, *args: object) -> None:
        if getattr(self, "path", "") == "/api/state" and len(args) > 1 and str(args[1]) == "200":
            return
        super().log_message(message_format, *args)

    def trusted(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in hosts or (origin and origin not in {f"http://{item}" for item in hosts}):
            self.respond(403, dict(error="This studio accepts same-origin localhost requests only"))
            return False
        return True

    def respond(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.headers_common(len(body))
        self.end_headers()
        self.wfile.write(body)

    def headers_common(self, length: int) -> None:
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                         "style-src 'self'; media-src 'self' https://huggingface.co https://*.huggingface.co "
                         "https://*.hf.co blob:; img-src 'self' data:; frame-ancestors 'none'")

    def serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.respond(404, dict(error="File not found"))
            return
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        request_range = self.headers.get("Range")
        if request_range:
            import re
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", request_range)
            if not match:
                self.respond(416, dict(error="Unsupported byte range"))
                return
            start = int(match[1])
            end = min(int(match[2]), end) if match[2] else end
            if not 0 <= start <= end:
                self.respond(416, dict(error="Invalid byte range"))
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.headers_common(end - start + 1)
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(65536, remaining))
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        if not self.trusted():
            return
        path = urlsplit(self.path).path
        if path == "/api/state":
            self.respond(200, self.server.studio.snapshot())
        elif path == "/api/examples":
            self.respond(200, examples())
        elif path == "/":
            self.serve_file(ASSETS / "index.html")
        elif path in ("/assets/app.js", "/assets/style.css"):
            self.serve_file(ASSETS / path.rsplit("/", 1)[1])
        elif path in self.server.studio.exports:
            self.serve_file(self.server.studio.exports[path])
        else:
            self.respond(404, dict(error="Not found"))

    def read_body(self, maximum: int) -> bytes:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked uploads are not supported")
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= maximum:
            raise ValueError(f"Request must contain 1..{maximum} bytes")
        self.connection.settimeout(30)
        content = self.rfile.read(size)
        if len(content) != size:
            raise ValueError("Incomplete upload")
        return content

    def upload_reference(self) -> dict:
        from fish_drama3.cli import reference_audio
        name = unquote(self.headers.get("X-Filename", "reference.wav"))
        suffix = Path(name).suffix.lower()
        if suffix not in (".wav", ".flac", ".mp3", ".ogg"):
            raise ValueError("Upload WAV, FLAC, MP3 or OGG reference audio")
        body = self.read_body(48 * 1024 * 1024)
        identifier = uuid.uuid4().hex
        folder = self.server.studio.output / "references"
        folder.mkdir(exist_ok=True)
        path = folder / (identifier + suffix)
        path.write_bytes(body)
        try:
            reference_audio(path)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise ValueError(f"Cannot read reference audio: {error}") from error
        self.server.studio.references[identifier] = path
        return dict(reference_id=identifier, name=Path(name).name)

    def sample_reference(self, data: dict) -> dict:
        import hashlib
        from urllib.request import urlopen
        from fish_drama3.cli import reference_audio

        if data.get("id") != "podcast":
            raise ValueError("No reference is available for this example")
        sample = next(item for item in examples()["examples"] if item["id"] == "podcast")
        # Fixed public fixture only: requests cannot choose a server-side URL.
        url = "https://github.com/user-attachments/files/33393249/en0.wav"
        with urlopen(url, timeout=30) as response:
            audio = response.read(16 * 1024 * 1024 + 1)
        if hashlib.sha256(audio).hexdigest() != "af3ac928e98cdb171c9abdc611813f2cc40c52b71167790b785628104f81f389":
            raise ValueError("Sample reference hash mismatch; download the reference manually")
        folder = self.server.studio.output / "references"
        folder.mkdir(exist_ok=True)
        identifier = uuid.uuid4().hex
        path = folder / (identifier + ".wav")
        path.write_bytes(audio)
        try:
            reference_audio(path)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise ValueError(f"Cannot read sample reference: {error}") from error
        self.server.studio.references[identifier] = path
        return dict(reference_id=identifier, name="en0.wav", reference_text=sample["reference_text"])

    def do_POST(self) -> None:
        if not self.trusted():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/reference":
                self.respond(200, self.upload_reference())
                return
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Send application/json")
            data = json.loads(self.read_body(128 * 1024))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            actions = {"/api/load": lambda: self.server.studio.load(data),
                       "/api/plan": lambda: self.server.studio.plan(data),
                       "/api/sample-reference": lambda: self.sample_reference(data),
                       "/api/generate": lambda: self.server.studio.generate(data),
                       "/api/unload": lambda: self.server.studio.stop(),
                       "/api/cancel": lambda: self.server.studio.stop("Cancelled. Reload the model to continue.")}
            if path not in actions:
                self.respond(404, dict(error="Not found"))
                return
            self.respond(202, actions[path]())
        except (ValueError, RuntimeError, OSError) as error:
            self.respond(400, dict(error=str(error)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/studio"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    studio = Studio(args.output_dir)
    studio.start_vitals()
    server = StudioServer(("127.0.0.1", args.port), studio)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Drama 3 Studio: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
