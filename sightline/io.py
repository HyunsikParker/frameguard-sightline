"""Bounded frame decoding.

Inputs are either a video file that OpenCV can decode or a directory of still images
(PNG/JPEG) that form a sequence. Limits are checked before and during decoding so that a
malformed or oversized upload is rejected with a reason instead of exhausting memory.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import InputLimits

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class InputRejected(ValueError):
    """Raised when an input breaks a limit or cannot be decoded."""


@dataclass(frozen=True)
class SourceInfo:
    path: str
    kind: str  # "video" or "image_sequence"
    frame_count: int
    width: int
    height: int
    fps: float


@dataclass
class Frame:
    index: int
    time_s: float
    bgr: np.ndarray


def _image_files(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise InputRejected(f"no PNG or JPEG frames in {directory}")
    return files


def _check_size(width: int, height: int, limits: InputLimits) -> None:
    if width <= 0 or height <= 0:
        raise InputRejected("frame has no pixels")
    if width > limits.max_width or height > limits.max_height:
        raise InputRejected(f"frame size {width}x{height} exceeds {limits.max_width}x{limits.max_height}")


def _total_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    return path.stat().st_size


def probe(path: str | os.PathLike, limits: InputLimits) -> SourceInfo:
    """Read metadata only and enforce byte, frame and size limits."""
    source = Path(path)
    if not source.exists():
        raise InputRejected(f"input not found: {source}")
    size = _total_bytes(source)
    if size > limits.max_input_bytes:
        raise InputRejected(f"input is {size} bytes, limit is {limits.max_input_bytes}")

    if source.is_dir():
        files = _image_files(source)
        if len(files) > limits.max_frames:
            raise InputRejected(f"{len(files)} frames, limit is {limits.max_frames}")
        first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise InputRejected(f"cannot decode {files[0].name}")
        height, width = first.shape[:2]
        _check_size(width, height, limits)
        return SourceInfo(str(source), "image_sequence", len(files), width, height, 24.0)

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise InputRejected(f"OpenCV cannot open {source.name} (unsupported codec or container)")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    finally:
        capture.release()
    _check_size(width, height, limits)
    if count > limits.max_frames:
        raise InputRejected(f"{count} frames, limit is {limits.max_frames}")
    return SourceInfo(str(source), "video", count, width, height, fps)


def iter_frames(
    info: SourceInfo, limits: InputLimits, start: int = 0, stop: int | None = None
) -> Iterator[Frame]:
    """Yield frames [start, stop) from a probed source, enforcing the frame limit again."""
    stop = info.frame_count if stop is None else min(stop, info.frame_count)
    if info.kind == "image_sequence":
        files = _image_files(Path(info.path))
        for index in range(start, stop):
            image = cv2.imread(str(files[index]), cv2.IMREAD_COLOR)
            if image is None:
                raise InputRejected(f"cannot decode frame {index} ({files[index].name})")
            _check_size(image.shape[1], image.shape[0], limits)
            yield Frame(index, index / info.fps, image)
        return

    capture = cv2.VideoCapture(info.path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, stop):
            ok, image = capture.read()
            if not ok or image is None:
                # A truncated container ends early; the caller records the shortfall.
                return
            yield Frame(index, index / info.fps, image)
    finally:
        capture.release()


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Downscale (never upscale) to the given width, keeping the aspect ratio."""
    height, current = image.shape[:2]
    if current <= width:
        return image
    new_height = max(1, round(height * width / current))
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
