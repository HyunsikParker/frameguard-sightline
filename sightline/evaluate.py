"""Score the agent against seeded ground truth.

A labelled defect counts as found when a ``defect`` or ``needs_review`` finding of the same
kind overlaps it. Findings with no matching label are false positives. Dismissed findings
are not scored as detections, but a dismissed finding that overlaps a label is a miss that
the report lists separately (the agent explained away a real defect).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .pipeline import analyze

DEFECT_KINDS = ["black_frame", "frozen_run", "flash_frame", "damaged_frame", "frame_skip"]


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int, slack: int = 1) -> bool:
    return a_start <= b_end + slack and b_start <= a_end + slack


def score_clip(clip_dir: Path, labels_path: Path) -> dict:
    truth = json.loads(labels_path.read_text())
    job = analyze(str(clip_dir))
    flagged = [f for f in job.report.findings if f.verdict in ("defect", "needs_review")]
    dismissed = [f for f in job.report.findings if f.verdict == "dismissed"]
    used = set()
    rows = []
    for label in truth["labels"]:
        match = next(
            (
                i
                for i, f in enumerate(flagged)
                if i not in used
                and f.kind == label["kind"]
                and _overlaps(f.start, f.end, label["start"], label["end"])
            ),
            None,
        )
        if match is not None:
            used.add(match)
        explained_away = (
            any(_overlaps(f.start, f.end, label["start"], label["end"]) for f in dismissed) and match is None
        )
        rows.append(
            {
                **label,
                "found": match is not None,
                "verdict": flagged[match].verdict if match is not None else None,
                "explained_away": explained_away,
            }
        )
    false_positives = [
        {"kind": f.kind, "start": f.start, "end": f.end, "verdict": f.verdict}
        for i, f in enumerate(flagged)
        if i not in used
    ]
    return {
        "clip": clip_dir.name,
        "frames": truth["frames"],
        "labels": rows,
        "false_positives": false_positives,
        "summary": job.summary(),
    }


def evaluate_set(directory: Path) -> dict:
    clips = sorted(
        p
        for p in directory.iterdir()
        if p.name.startswith("clip_") and (p.is_dir() or p.suffix.lower() in (".mp4", ".mov"))
    )
    per_clip = [score_clip(c, directory / f"{c.stem if c.is_file() else c.name}.labels.json") for c in clips]
    totals = defaultdict(lambda: {"labels": 0, "found": 0, "found_as_defect": 0, "false_positives": 0})
    for clip in per_clip:
        for row in clip["labels"]:
            t = totals[row["kind"]]
            t["labels"] += 1
            t["found"] += row["found"]
            t["found_as_defect"] += row["verdict"] == "defect"
        for fp in clip["false_positives"]:
            totals[fp["kind"]]["false_positives"] += 1
    by_kind = {}
    for kind, t in totals.items():
        flagged = t["found"] + t["false_positives"]
        by_kind[kind] = {
            **t,
            "recall": round(t["found"] / t["labels"], 3) if t["labels"] else None,
            "precision": round(t["found"] / flagged, 3) if flagged else None,
        }
    frames = sum(c["frames"] for c in per_clip)
    return {
        "clips": len(per_clip),
        "frames": frames,
        "by_kind": by_kind,
        "false_positives_per_1000_frames": round(
            1000 * sum(t["false_positives"] for t in totals.values()) / max(frames, 1), 2
        ),
        "mean_ms_per_frame_first_pass": round(
            sum(c["summary"]["ms_per_frame_first_pass"] for c in per_clip) / max(len(per_clip), 1), 2
        ),
        "mean_tool_calls": round(
            sum(c["summary"]["tool_calls"] for c in per_clip) / max(len(per_clip), 1), 1
        ),
        "per_clip": per_clip,
    }
