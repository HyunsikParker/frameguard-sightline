"""Write a job's results: findings, trace, thumbnails and a human-readable summary.

Thumbnails are small JPEGs of the frames named in each finding, so an editor can check a
claim without opening the source. No other image content leaves the job directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .io import iter_frames, resize_to_width
from .pipeline import JobResult

VERDICT_ORDER = {"defect": 0, "needs_review": 1, "dismissed": 2}


def _evidence_frames(start: int, end: int, frame_count: int) -> list[int]:
    """The frame before, the first and last affected frames, and the frame after."""
    picks = [start - 1, start, end, end + 1]
    return sorted({i for i in picks if 0 <= i < frame_count})


def write_thumbnails(job: JobResult, out_dir: Path) -> dict[int, str]:
    wanted: set[int] = set()
    for f in job.report.findings:
        if f.verdict != "dismissed":
            wanted.update(_evidence_frames(f.start, f.end, job.info.frame_count))
    thumbs: dict[int, str] = {}
    if not wanted:
        return thumbs
    thumb_dir = out_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for frame in iter_frames(job.info, job.settings.limits, min(wanted), max(wanted) + 1):
        if frame.index in wanted:
            name = f"frame_{frame.index:05d}.jpg"
            small = resize_to_width(frame.bgr, job.settings.limits.thumb_width)
            cv2.imwrite(str(thumb_dir / name), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumbs[frame.index] = f"thumbs/{name}"
    return thumbs


def contact_sheet(out_dir: Path, thumbs: dict[int, str], job: JobResult) -> str | None:
    """One strip per flagged finding: before | first | last | after, labelled."""
    rows = []
    for f in sorted(job.report.findings, key=lambda f: f.start):
        if f.verdict == "dismissed":
            continue
        tiles = []
        for i in _evidence_frames(f.start, f.end, job.info.frame_count):
            path = thumbs.get(i)
            if path is None:
                continue
            tile = cv2.imread(str(out_dir / path))
            cv2.putText(tile, str(i), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        if not tiles:
            continue
        height = max(t.shape[0] for t in tiles)
        tiles = [
            cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            for t in tiles
        ]
        strip = np.hstack(tiles)
        label = np.full((26, strip.shape[1], 3), 32, dtype=np.uint8)
        cv2.putText(
            label,
            f"{f.kind} {f.start}-{f.end} {f.verdict}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([label, strip]))
    if not rows:
        return None
    width = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 6, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
        for r in rows
    ]
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "contact_sheet.jpg"


def summary_markdown(job: JobResult) -> str:
    s = job.summary()
    lines = [
        f"# Sightline report: {Path(job.info.path).name}",
        "",
        f"{s['frames_scanned']} frames at {job.info.width}x{job.info.height}, OpenCV {s['opencv_version']}. "
        f"First pass {s['ms_per_frame_first_pass']} ms/frame; "
        f"{s['tool_calls']} follow-up checks re-decoded {s['redecoded_frames']} frames.",
        "",
        f"**{s['defects']} defects**, {s['needs_review']} for review, "
        f"{s['dismissed']} dismissed as intended material.",
        "",
        "| Frames | Kind | Verdict | Confidence | Why | Proposed action |",
        "|---|---|---|---|---|---|",
    ]
    for f in sorted(job.report.findings, key=lambda f: (VERDICT_ORDER[f.verdict], f.start)):
        action = f.proposed_action or ""
        lines.append(
            f"| {f.start}-{f.end} | {f.kind} | {f.verdict} | {f.confidence:.2f} | {f.reason} | {action} |"
        )
    if job.report.budget_exhausted:
        lines += ["", "The follow-up budget ran out; unchecked candidates are listed for review."]
    return "\n".join(lines) + "\n"


def write_report(job: JobResult, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = write_thumbnails(job, out_dir)
    sheet = contact_sheet(out_dir, thumbs, job)
    record = {"summary": job.summary(), **job.report.to_dict(), "thumbnails": thumbs, "contact_sheet": sheet}
    (out_dir / "findings.json").write_text(json.dumps(record, indent=1, default=float))
    (out_dir / "summary.md").write_text(summary_markdown(job))
    return record
