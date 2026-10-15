"""First pass: measure every frame once at analysis resolution.

The pass keeps only numbers (no images) so that memory stays flat for long clips. Later
tool calls re-decode the small windows they need at full resolution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2

from .config import InputLimits
from .io import SourceInfo, iter_frames, resize_to_width
from .measures import (
    FrameStats,
    flow_field,
    flow_magnitude,
    frame_stats,
    hamming,
    histogram_distance,
    median_flow_magnitude,
)

CHANGE_LEVEL = 6  # grey levels; below this a pixel difference is treated as codec noise


@dataclass
class ScanResult:
    info: SourceInfo
    stats: list[FrameStats]
    # Pair series: element i compares frame i-1 with frame i; element 0 is NaN.
    flow: list[float] = field(default_factory=list)  # trimmed mean magnitude
    flow_median: list[float] = field(default_factory=list)  # whole-frame motion
    hist_distance: list[float] = field(default_factory=list)
    hash_distance: list[float] = field(default_factory=list)
    # Fraction of pixels whose grey level changes by more than CHANGE_LEVEL between the two frames.
    changed_fraction: list[float] = field(default_factory=list)
    decoded_frames: int = 0
    seconds: float = 0.0

    @property
    def frame_count(self) -> int:
        return len(self.stats)


def scan(info: SourceInfo, limits: InputLimits) -> ScanResult:
    started = time.perf_counter()
    result = ScanResult(info=info, stats=[])
    previous_gray = None
    previous = None
    for frame in iter_frames(info, limits):
        small = resize_to_width(frame.bgr, limits.analysis_width)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        stats = frame_stats(frame.index, small)
        if previous is None:
            result.flow.append(math.nan)
            result.flow_median.append(math.nan)
            result.hist_distance.append(math.nan)
            result.hash_distance.append(math.nan)
            result.changed_fraction.append(math.nan)
        else:
            flow = flow_field(previous_gray, gray)
            result.flow.append(flow_magnitude(flow))
            result.flow_median.append(median_flow_magnitude(flow))
            result.changed_fraction.append(float((cv2.absdiff(previous_gray, gray) > CHANGE_LEVEL).mean()))
            result.hist_distance.append(histogram_distance(previous.hist, stats.hist))
            result.hash_distance.append(float(hamming(previous.phash, stats.phash)))
        result.stats.append(stats)
        previous, previous_gray = stats, gray
        result.decoded_frames += 1
    result.seconds = time.perf_counter() - started
    return result
