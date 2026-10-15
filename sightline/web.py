"""Local stand-in for the AWS deployment: same review page, same JSON API, no cloud.

    python -m sightline.web --port 8765 --data jobs-local

Uploads, reports and previews stay in the data directory. The workflow (analyze, wait for a
person, act and re-check) runs in a background thread instead of Step Functions.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import uuid
from email.parser import BytesParser
from email.policy import default as default_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .actions import apply_approved, record_decision
from .pipeline import analyze
from .report import write_report

PAGE = Path(__file__).resolve().parent.parent / "aws" / "api" / "index.html"
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
JOB = r"([0-9a-f]{32})"


class Store:
    """Job records in memory, files on disk."""

    def __init__(self, root: Path):
        self.root = root
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def update(self, job_id: str, **fields) -> None:
        with self.lock:
            self.jobs[job_id].update(fields)

    def run_analysis(self, job_id: str) -> None:
        job = self.jobs[job_id]
        try:
            self.update(job_id, status="analyzing")
            source = self.job_dir(job_id) / job["input_name"]
            record = write_report(analyze(str(source)), self.job_dir(job_id) / "report")
            pending = sum(f["approval"] == "pending" for f in record["findings"])
            self.update(job_id, status="awaiting_approval" if pending else "done", pending=pending)
        except Exception as error:  # surfaced to the page as a failed job
            self.update(job_id, status="failed", error=str(error))

    def run_actions(self, job_id: str, decisions: list[dict]) -> None:
        job = self.jobs[job_id]
        try:
            report = self.job_dir(job_id) / "report"
            for d in decisions:
                record_decision(report, d["finding"], d["decision"], d["by"])
            result = apply_approved(report, str(self.job_dir(job_id) / job["input_name"]))
            self.update(job_id, status="done", actions=result)
        except Exception as error:
            self.update(job_id, status="failed", error=str(error))


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            return

        def _send(self, status: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, data: bytes, ctype: str) -> None:
            """Serve a file or one byte range of it, so the browser can seek in the repair preview."""
            size = len(data)
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("range", "").strip())
            if not m or not (m.group(1) or m.group(2)):
                return self._send(200, data, ctype, {"accept-ranges": "bytes"})
            if m.group(1):
                first, last = int(m.group(1)), min(int(m.group(2) or size - 1), size - 1)
            else:  # a suffix range: the last N bytes
                first, last = max(0, size - int(m.group(2))), size - 1
            if first > last:
                return self._send(416, b"", ctype, {"content-range": f"bytes */{size}"})
            headers = {"accept-ranges": "bytes", "content-range": f"bytes {first}-{last}/{size}"}
            return self._send(206, data[first : last + 1], ctype, headers)

        def _json(self, status: int, body: dict) -> None:
            self._send(status, json.dumps(body).encode(), "application/json")

        def _body(self) -> bytes:
            length = int(self.headers.get("content-length", "0"))
            if length > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise ValueError("request too large")
            return self.rfile.read(length)

        def do_GET(self):
            self.path = urlsplit(self.path).path  # the page takes ?job=<id>; routing ignores the query
            if self.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            m = re.fullmatch(rf"/api/jobs/{JOB}", self.path)
            if m and m.group(1) in store.jobs:
                return self._json(200, self._job_view(m.group(1)))
            m = re.fullmatch(rf"/local-media/{JOB}/([A-Za-z0-9_./-]+)", self.path)
            if m and ".." not in m.group(2):
                path = store.job_dir(m.group(1)) / "report" / m.group(2)
                if path.is_file():
                    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    return self._send_media(path.read_bytes(), ctype)
            return self._json(404, {"error": "not found"})

        def _job_view(self, job_id: str) -> dict:
            job = dict(store.jobs[job_id])
            out = {"job": job}
            findings = store.job_dir(job_id) / "report" / "findings.json"
            if findings.exists():
                record = json.loads(findings.read_text())
                media = f"/local-media/{job_id}"
                out.update(
                    {"findings": record["findings"], "trace": record["trace"], "summary": record["summary"]}
                )
                out["media"] = {
                    "contact_sheet": f"{media}/contact_sheet.jpg" if record.get("contact_sheet") else None,
                    "thumbs": {k: f"{media}/{v}" for k, v in record.get("thumbnails", {}).items()},
                }
                if job.get("actions", {}).get("preview"):
                    out["media"]["repair_preview"] = f"{media}/{job['actions']['preview']}"
            return out

        def do_POST(self):
            try:
                if self.path == "/api/jobs":
                    body = json.loads(self._body() or b"{}")
                    name = Path(str(body.get("filename", ""))).name
                    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                        return self._json(400, {"error": "upload a video file"})
                    if not 0 < int(body.get("size", 0)) <= MAX_UPLOAD_BYTES:
                        return self._json(400, {"error": "file is empty or larger than 80 MB"})
                    job_id = uuid.uuid4().hex
                    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80]
                    store.job_dir(job_id).mkdir(parents=True)
                    store.jobs[job_id] = {"job_id": job_id, "status": "created", "input_name": safe}
                    return self._json(
                        201, {"job_id": job_id, "upload": {"url": f"/local-upload/{job_id}", "fields": {}}}
                    )
                m = re.fullmatch(rf"/local-upload/{JOB}", self.path)
                if m and m.group(1) in store.jobs:
                    raw = self._body()
                    message = BytesParser(policy=default_policy).parsebytes(
                        b"content-type: " + self.headers["content-type"].encode() + b"\r\n\r\n" + raw
                    )
                    for part in message.iter_parts():
                        if part.get_param("name", header="content-disposition") == "file":
                            target = store.job_dir(m.group(1)) / store.jobs[m.group(1)]["input_name"]
                            target.write_bytes(part.get_payload(decode=True))
                            return self._send(204, b"", "text/plain")
                    return self._json(400, {"error": "no file in upload"})
                m = re.fullmatch(rf"/api/jobs/{JOB}/start", self.path)
                if m and store.jobs.get(m.group(1), {}).get("status") == "created":
                    store.update(m.group(1), status="queued")
                    threading.Thread(target=store.run_analysis, args=(m.group(1),), daemon=True).start()
                    return self._json(202, {"job_id": m.group(1), "status": "queued"})
                m = re.fullmatch(rf"/api/jobs/{JOB}/decisions", self.path)
                if m and store.jobs.get(m.group(1), {}).get("status") == "awaiting_approval":
                    body = json.loads(self._body() or b"{}")
                    by = re.sub(r"[^A-Za-z0-9 ._@-]", "", str(body.get("by", "reviewer")))[:40] or "reviewer"
                    decisions = [
                        {"finding": str(d.get("finding")), "decision": d.get("decision"), "by": by}
                        for d in body.get("decisions", [])
                        if d.get("decision") in ("approve", "reject")
                    ]
                    if not decisions:
                        return self._json(400, {"error": "no decisions given"})
                    store.update(m.group(1), status="acting", decisions=decisions)
                    threading.Thread(
                        target=store.run_actions, args=(m.group(1), decisions), daemon=True
                    ).start()
                    return self._json(202, {"job_id": m.group(1), "status": "acting"})
                return self._json(409, {"error": "not allowed in the job's current state"})
            except (ValueError, json.JSONDecodeError) as error:
                return self._json(400, {"error": str(error)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", default="jobs-local")
    args = parser.parse_args()
    store = Store(Path(args.data))
    store.root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Sightline review page on http://127.0.0.1:{args.port}/ (data in {store.root})")
    server.serve_forever()


if __name__ == "__main__":
    main()
