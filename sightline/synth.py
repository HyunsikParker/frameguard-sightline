"""Synthetic test clips with seeded defects and ground-truth labels.

Every clip is generated from a seed, so the public evaluation set can be rebuilt exactly.
Clips mix material that must NOT be flagged (intended cuts, fades, a static title card,
slow motion) with injected defects that must be flagged.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 640, 360, 24


@dataclass
class Label:
    kind: str
    start: int
    end: int


PAD = 200  # texture margin so the background can drift without wrapping


def _texture(rng: np.random.Generator, scale: int) -> np.ndarray:
    noise = rng.integers(
        0, 255, ((HEIGHT + 2 * PAD) // scale + 2, (WIDTH + 2 * PAD) // scale + 2), dtype=np.uint8
    )
    return cv2.resize(noise, (WIDTH + 2 * PAD, HEIGHT + 2 * PAD), interpolation=cv2.INTER_CUBIC)


class MovingScene:
    """A textured background drifting under moving shapes; `speed` scales all motion."""

    def __init__(self, seed: int, speed: float = 1.0):
        rng = np.random.default_rng(seed)
        self.palette = rng.integers(40, 220, size=(3, 3)).astype(np.float32)
        self.texture = _texture(rng, scale=int(rng.integers(6, 14)))
        self.shapes = [
            {
                "pos": rng.uniform([60, 60], [WIDTH - 60, HEIGHT - 60]),
                "vel": rng.uniform(-3.5, 3.5, size=2) * speed,
                "radius": int(rng.integers(18, 46)),
                "color": tuple(int(c) for c in rng.integers(0, 255, size=3)),
                "square": bool(rng.integers(0, 2)),
            }
            for _ in range(int(rng.integers(3, 6)))
        ]
        self.drift = rng.uniform(-1.2, 1.2, size=2) * speed
        self.t = 0

    def next(self) -> np.ndarray:
        # Continuous drift, clamped inside the padded texture (no wrap-around jumps).
        dx = int(np.clip(PAD + round(self.drift[0] * self.t), 0, 2 * PAD))
        dy = int(np.clip(PAD + round(self.drift[1] * self.t), 0, 2 * PAD))
        tex = self.texture[dy : dy + HEIGHT, dx : dx + WIDTH].astype(np.float32) / 255.0
        grad = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)[None, :, None]
        base = self.palette[0] * (1 - grad) + self.palette[1] * grad
        frame = base * (0.65 + 0.35 * tex[..., None])
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        for shape in self.shapes:
            shape["pos"] += shape["vel"]
            for axis, bound in ((0, WIDTH), (1, HEIGHT)):
                if not shape["radius"] <= shape["pos"][axis] <= bound - shape["radius"]:
                    shape["vel"][axis] *= -1
            x, y = (int(v) for v in shape["pos"])
            r = shape["radius"]
            if shape["square"]:
                cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), shape["color"], -1)
            else:
                cv2.circle(frame, (x, y), r, shape["color"], -1, lineType=cv2.LINE_AA)
        self.t += 1
        return frame


def title_card(text: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    color = tuple(int(c) for c in rng.integers(30, 90, size=3))
    frame = np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)
    cv2.putText(
        frame, text, (60, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (235, 235, 235), 3, cv2.LINE_AA
    )
    return frame


def corrupt_blocks(frame: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    """Mimic a damaged export: misplaced 16x16 macroblocks with flat or shifted content."""
    out = frame.copy()
    blocks_y, blocks_x = HEIGHT // 16, WIDTH // 16
    count = max(1, int(blocks_x * blocks_y * fraction))
    for _ in range(count):
        by, bx = int(rng.integers(0, blocks_y)), int(rng.integers(0, blocks_x))
        y, x = by * 16, bx * 16
        if rng.random() < 0.5:
            out[y : y + 16, x : x + 16] = rng.integers(0, 255, size=3)
        else:
            sy, sx = int(rng.integers(0, blocks_y)) * 16, int(rng.integers(0, blocks_x)) * 16
            out[y : y + 16, x : x + 16] = frame[sy : sy + 16, sx : sx + 16]
    return out


def build_clip(seed: int) -> tuple[list[np.ndarray], list[Label]]:
    """One clip: three moving shots, a title card, a fade, and 3-5 injected defects."""
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    frames: list[np.ndarray] = []
    labels: list[Label] = []

    def shot(scene: MovingScene, n: int, defects: list[str]) -> None:
        """Append n frames of a moving scene and inject the requested defects inside it."""
        planned = sorted(pyr.sample(range(14, n - 18), k=len(defects))) if defects else []
        plan = dict(zip(planned, defects, strict=False))
        i = 0
        while i < n:
            kind = plan.get(i)
            if kind == "black_frame":
                length = pyr.randint(1, 3)
                start = len(frames)
                for _ in range(length):
                    frames.append(np.clip(rng.normal(3, 1.5, (HEIGHT, WIDTH, 3)), 0, 255).astype(np.uint8))
                labels.append(Label(kind, start, start + length - 1))
                i += 1
            elif kind == "frozen_run":
                length = pyr.randint(8, 14)
                held = scene.next()
                start = len(frames)
                frames.extend(held.copy() for _ in range(length))
                labels.append(Label(kind, start, start + length - 1))
                i += 1
            elif kind == "flash_frame":
                length = pyr.randint(1, 2)
                intruder = MovingScene(seed * 31 + i)
                start = len(frames)
                for _ in range(length):
                    frames.append(intruder.next())
                    scene.next()  # the real shot keeps time underneath the flash
                labels.append(Label(kind, start, start + length - 1))
                i += length
            elif kind == "damaged_frame":
                length = pyr.randint(1, 2)
                start = len(frames)
                for _ in range(length):
                    frames.append(corrupt_blocks(scene.next(), rng, fraction=pyr.uniform(0.06, 0.18)))
                labels.append(Label(kind, start, start + length - 1))
                i += length
            elif kind == "frame_skip":
                for _ in range(pyr.randint(5, 8)):
                    scene.next()  # frames dropped from the export
                frames.append(scene.next())
                labels.append(Label(kind, len(frames) - 1, len(frames) - 1))
                i += 1
            else:
                frames.append(scene.next())
                i += 1

    kinds = ["black_frame", "frozen_run", "flash_frame", "damaged_frame", "frame_skip"]
    chosen = pyr.sample(kinds, k=pyr.randint(3, 5))
    split = [chosen[: len(chosen) // 2], chosen[len(chosen) // 2 :]]

    shot(MovingScene(seed * 7 + 1), 72, split[0])
    # Intended hard cut to a second moving shot (not a defect).
    shot(MovingScene(seed * 7 + 2, speed=1.3), 64, split[1])
    # Intended cut to a static title card held for 30 frames (static, but not a freeze defect).
    card = title_card(f"Scene {seed % 97}", seed)
    frames.extend(card.copy() for _ in range(30))
    # Slow but real motion (must not be called frozen), then a fade to black (not a defect).
    slow = MovingScene(seed * 7 + 3, speed=0.2)
    shot(slow, 36, [])
    last = frames[-1]
    for k in range(1, 13):
        frames.append((last.astype(np.float32) * (1 - k / 12)).astype(np.uint8))
    return frames, labels


def write_clip(directory: Path, seed: int) -> Path:
    frames, labels = build_clip(seed)
    clip_dir = directory / f"clip_{seed:04d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(clip_dir / f"{index:05d}.png"), frame)
    (directory / f"clip_{seed:04d}.labels.json").write_text(
        json.dumps(
            {"seed": seed, "frames": len(frames), "fps": FPS, "labels": [asdict(label) for label in labels]},
            indent=1,
        )
    )
    return clip_dir


def write_set(directory: Path, seeds: list[int]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    return [write_clip(directory, seed) for seed in seeds]


def write_video(frames: list[np.ndarray], path: Path, fps: int = FPS) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path
