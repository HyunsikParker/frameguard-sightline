"""Seed defects into openly licensed real footage to measure recall on real edits.

The clean trailer is scanned first so that every defect lands inside a moving shot, away
from the trailer's own cuts and fades. The seeded copy is written as an MP4 (a compressed
delivery file, like a real export) together with frame-accurate labels.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_SETTINGS
from .io import iter_frames, probe, resize_to_width
from .scan import scan
from .synth import Label

KINDS = ["black_frame", "frozen_run", "flash_frame", "damaged_frame", "frame_skip"]


def _eligible_positions(source: str, margin: int = 10) -> list[int]:
    """Frames inside moving shots: no shot change or dark frame within `margin` frames."""
    info = probe(source, DEFAULT_SETTINGS.limits)
    first = scan(info, DEFAULT_SETTINGS.limits)
    t = DEFAULT_SETTINGS.thresholds
    unsafe = set()
    for i in range(1, first.frame_count):
        if first.hist_distance[i] >= t.shot_stable_hist_distance or first.stats[i].luma_mean < 40:
            unsafe.update(range(i - margin, i + margin + 1))
    positions = []
    for i in range(margin, first.frame_count - margin):
        flows = [f for f in first.flow[i - 3 : i + 4] if not np.isnan(f)]
        if i not in unsafe and flows and float(np.median(flows)) > 0.6:
            positions.append(i)
    return positions


def _corrupt(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for _ in range(int(rng.integers(25, 60))):
        y, x = int(rng.integers(0, h // 16)) * 16, int(rng.integers(0, w // 16)) * 16
        out[y : y + 16, x : x + 16] = rng.integers(0, 255, size=3)
    return out


def seed_video(source: str, out_dir: Path, seed: int, width: int = 640) -> Path:
    """Write clip_<seed>.mp4 with one defect of each kind and clip_<seed>.labels.json."""
    pyr = random.Random(seed)
    rng = np.random.default_rng(seed)
    positions = _eligible_positions(source)
    chosen: list[int] = []
    for pos in pyr.sample(positions, k=len(positions)):
        if all(abs(pos - other) > 40 for other in chosen):
            chosen.append(pos)
        if len(chosen) == len(KINDS):
            break
    plan = dict(zip(sorted(chosen), pyr.sample(KINDS, k=len(chosen)), strict=False))

    info = probe(source, DEFAULT_SETTINGS.limits)
    limits = DEFAULT_SETTINGS.limits
    # Frames from another part of the trailer stand in for an accidental insert (a flash).
    far_index = {
        i: (i + info.frame_count // 2) % info.frame_count for i, k in plan.items() if k == "flash_frame"
    }
    wanted = set(far_index.values())
    far = {f.index: resize_to_width(f.bgr, width) for f in iter_frames(info, limits) if f.index in wanted}

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"clip_{seed:04d}.mp4"
    writer = None
    written = 0
    labels: list[Label] = []
    hold: tuple[np.ndarray, int] | None = None  # frame being held for a frozen run, frames left
    skip_left = 0

    def emit(frame: np.ndarray) -> None:
        nonlocal writer, written
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (w, h))
        writer.write(frame)
        written += 1

    for f in iter_frames(info, limits):
        frame = resize_to_width(f.bgr, width)
        if skip_left:
            skip_left -= 1
            if skip_left == 0:
                labels.append(Label("frame_skip", written, written))
                emit(frame)
            continue
        if hold is not None:
            held, left = hold
            emit(held.copy())
            hold = (held, left - 1) if left > 1 else None
            continue
        kind = plan.get(f.index)
        if kind == "black_frame":
            n = pyr.randint(1, 2)
            labels.append(Label(kind, written, written + n - 1))
            for _ in range(n):
                emit(np.zeros_like(frame))
            emit(frame)
        elif kind == "frozen_run":
            n = pyr.randint(8, 12)
            labels.append(Label(kind, written, written + n - 1))
            emit(frame)
            hold = (frame, n - 1)  # the held frame replaces the next n-1 source frames
        elif kind == "flash_frame":
            labels.append(Label(kind, written, written))
            emit(far[far_index[f.index]])
        elif kind == "damaged_frame":
            labels.append(Label(kind, written, written))
            emit(_corrupt(frame, rng))
        elif kind == "frame_skip":
            skip_left = pyr.randint(4, 6)  # drop these source frames, label the next one written
        else:
            emit(frame)
    if writer is not None:
        writer.release()

    (out_dir / f"clip_{seed:04d}.labels.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "source": Path(source).name,
                "frames": written,
                "fps": info.fps,
                "labels": [asdict(label) for label in labels],
            },
            indent=1,
        )
    )
    return path
