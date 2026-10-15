"""Human approval and the actions it unlocks.

The agent never changes footage on its own. A finding with ``approval == "pending"`` waits
for a person. Approved repairable findings (black, flash and damaged frames) are patched in
a *preview* copy by holding the nearest good frame of the same shot; frozen runs and frame
skips become entries in a re-render list, because they cannot be repaired from the export.
Every decision and action is appended to ``actions.json`` with its time and author.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .agent import REPAIRABLE
from .config import DEFAULT_SETTINGS, InputLimits
from .io import iter_frames, probe


class ApprovalError(ValueError):
    pass


def finding_id(finding: dict) -> str:
    return f"{finding['kind']}:{finding['start']}-{finding['end']}"


def _load(job_dir: Path) -> dict:
    return json.loads((job_dir / "findings.json").read_text())


def _append_log(job_dir: Path, entry: dict) -> None:
    log_path = job_dir / "actions.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry})
    log_path.write_text(json.dumps(log, indent=1))


def record_decision(job_dir: Path, fid: str, decision: str, by: str, note: str = "") -> dict:
    """Store a person's decision on one pending finding."""
    if decision not in ("approve", "reject"):
        raise ApprovalError("decision must be 'approve' or 'reject'")
    record = _load(job_dir)
    for finding in record["findings"]:
        if finding_id(finding) == fid:
            if finding["approval"] != "pending":
                raise ApprovalError(f"{fid} is not waiting for approval (state: {finding['approval']})")
            finding["approval"] = "approved" if decision == "approve" else "rejected"
            finding["decided_by"] = by
            (job_dir / "findings.json").write_text(json.dumps(record, indent=1))
            _append_log(
                job_dir, {"event": "decision", "finding": fid, "decision": decision, "by": by, "note": note}
            )
            return finding
    raise ApprovalError(f"no finding {fid}")


# Browsers play H.264 MP4 and VP8 WebM but not MPEG-4 Part 2, so try those first.
PREVIEW_CODECS = (("avc1", ".mp4"), ("VP80", ".webm"), ("mp4v", ".mp4"))
# The preview confirms a diagnosis; full resolution is the re-render's job. Capping the width
# keeps encoding time inside the Lambda limit for 4K input.
PREVIEW_MAX_WIDTH = 960


def _open_preview(job_dir: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    for fourcc, suffix in PREVIEW_CODECS:
        path = job_dir / f"repair_preview{suffix}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("no video encoder available for the repair preview")


def _replacement_index(start: int, end: int, frame_count: int) -> int:
    """Prefer the frame just before the range; fall back to the one after it."""
    return start - 1 if start > 0 else min(end + 1, frame_count - 1)


def apply_approved(job_dir: Path, source: str, limits: InputLimits = DEFAULT_SETTINGS.limits) -> dict:
    """Run the actions unlocked by approvals: a repair preview and a re-render list."""
    record = _load(job_dir)
    approved = [f for f in record["findings"] if f["approval"] == "approved"]
    repairs = [f for f in approved if f["kind"] in REPAIRABLE]
    rerender = [f for f in approved if f["kind"] not in REPAIRABLE]
    result: dict = {"repaired_ranges": [], "rerender_ranges": [], "preview": None}

    if rerender:
        with (job_dir / "rerender_list.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kind", "first_frame", "last_frame", "reason"])
            for f in rerender:
                writer.writerow([f["kind"], f["start"], f["end"], f["reason"]])
                result["rerender_ranges"].append(finding_id(f))

    if repairs:
        info = probe(source, limits)
        replace = {}
        for f in repairs:
            source_index = _replacement_index(f["start"], f["end"], info.frame_count)
            for i in range(f["start"], f["end"] + 1):
                replace[i] = source_index
            result["repaired_ranges"].append(finding_id(f))
        size = (info.width, info.height)
        if info.width > PREVIEW_MAX_WIDTH:
            size = (PREVIEW_MAX_WIDTH, round(info.height * PREVIEW_MAX_WIDTH / info.width / 2) * 2)
        writer, preview_path = _open_preview(job_dir, info.fps, size)
        held: dict[int, np.ndarray] = {}
        needed = set(replace.values())
        for frame in iter_frames(info, limits):
            if frame.index in needed:
                held[frame.index] = frame.bgr.copy()
            out = held.get(replace[frame.index]) if frame.index in replace else frame.bgr
            out = frame.bgr if out is None else out
            if out.shape[1] != size[0]:
                out = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
            writer.write(out)
        writer.release()
        result["preview"] = preview_path.name

    _append_log(job_dir, {"event": "actions_applied", **result})
    if result["preview"]:
        result["verification"] = verify_repairs(job_dir, repairs)
    return result


def verify_repairs(job_dir: Path, repairs: list[dict]) -> dict:
    """Close the loop: re-analyse the repair preview and check each repaired range is clean."""
    from .pipeline import analyze  # local import keeps the module light for approval-only use

    preview = next(job_dir.glob("repair_preview.*"))
    job = analyze(str(preview))
    remaining = []
    for f in repairs:
        still_flagged = [
            g
            for g in job.report.findings
            if g.verdict != "dismissed"
            and g.kind == f["kind"]
            and g.start <= f["end"] + 1
            and f["start"] <= g.end + 1
        ]
        if still_flagged:
            remaining.append(finding_id(f))
    outcome = {
        "checked": [finding_id(f) for f in repairs],
        "still_flagged": remaining,
        "clean": not remaining,
    }
    _append_log(job_dir, {"event": "repairs_verified", **outcome})
    return outcome
