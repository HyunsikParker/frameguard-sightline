"""Turn first-pass measurements into candidate events.

A candidate is a suspicion, not a verdict. Each carries the measurements that raised it
and a margin that tells the agent how far past its threshold the evidence sits. The agent
decides which follow-up check each candidate needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import Thresholds
from .scan import ScanResult

BLACK = "black_frame"
FREEZE = "frozen_run"
CUT = "shot_change"
DAMAGE = "damaged_frame"
SKIP = "frame_skip"


@dataclass
class Candidate:
    kind: str
    start: int
    end: int  # inclusive
    margin: float  # >= 1.0 means past the threshold; larger is stronger
    measurements: dict[str, float] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    runs, start = [], None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _robust_z(values: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD), with a floor so flat clips do not divide by zero."""
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    return (values - median) / max(mad, 1e-3)


def find_black(scan: ScanResult, t: Thresholds) -> list[Candidate]:
    flags = [s.luma_mean < t.black_luma_mean and s.luma_std < t.black_luma_std for s in scan.stats]
    candidates = []
    for start, end in _runs(flags):
        run = scan.stats[start : end + 1]
        darkest = min(s.luma_mean for s in run)
        before = scan.stats[start - 1].luma_mean if start > 0 else math.nan
        drop = before - run[0].luma_mean if start > 0 else math.nan
        margin = (t.black_luma_mean / max(darkest, 1.0)) if darkest > 0 else 10.0
        candidates.append(
            Candidate(BLACK, start, end, margin, {"darkest_luma": darkest, "luma_drop_into_run": drop})
        )
    return candidates


def find_freezes(scan: ScanResult, t: Thresholds, black_frames: set[int]) -> list[Candidate]:
    flags = [False]
    for i in range(1, scan.frame_count):
        still = (
            scan.hash_distance[i] <= t.freeze_hash_distance
            and scan.changed_fraction[i] < t.freeze_changed_fraction
            and i not in black_frames
        )
        flags.append(still)
    candidates = []
    for start, end in _runs(flags):
        # Pair i compares i-1 with i, so the frozen picture starts one frame earlier.
        first = start - 1
        if end - first + 1 < t.freeze_min_frames:
            continue
        flows = scan.flow[start : end + 1]
        changed = scan.changed_fraction[start : end + 1]
        margin = t.freeze_changed_fraction / max(max(changed), 1e-5)
        candidates.append(
            Candidate(
                FREEZE,
                first,
                end,
                min(margin, 10.0),
                {"max_flow_px": float(max(flows)), "frames": end - first + 1},
            )
        )
    return candidates


def find_shot_changes(scan: ScanResult, t: Thresholds) -> list[Candidate]:
    candidates = []
    for i in range(1, scan.frame_count):
        hist_d = scan.hist_distance[i]
        previous, current = scan.stats[i - 1], scan.stats[i]
        edge_change = abs(current.edge_density - previous.edge_density) / max(
            current.edge_density, previous.edge_density, 1e-3
        )
        hash_d = scan.hash_distance[i]
        score = max(hist_d / t.cut_hist_distance, hash_d / 24.0)
        if score >= 1.0 and (edge_change >= t.cut_edge_change or hist_d >= t.cut_hist_distance):
            candidates.append(
                Candidate(
                    CUT,
                    i,
                    i,
                    score,
                    {"hist_distance": hist_d, "hash_distance": hash_d, "edge_change": edge_change},
                )
            )
    return candidates


def find_damage(scan: ScanResult, t: Thresholds) -> list[Candidate]:
    ratios = np.array([s.block_ratio for s in scan.stats], dtype=np.float64)
    z = _robust_z(ratios)
    flags = [
        bool(
            ratio >= t.damage_block_ratio
            and score >= t.damage_zscore
            and stats.block_step >= t.damage_min_step
        )
        for ratio, score, stats in zip(ratios, z, scan.stats, strict=True)
    ]
    candidates = []
    for start, end in _runs(flags):
        peak = float(ratios[start : end + 1].max())
        candidates.append(
            Candidate(
                DAMAGE,
                start,
                end,
                peak / t.damage_block_ratio,
                {
                    "peak_block_ratio": peak,
                    "clip_median_ratio": float(np.median(ratios)),
                    "peak_z": float(z[start : end + 1].max()),
                },
            )
        )
    return candidates


def find_skips(scan: ScanResult, t: Thresholds, cut_frames: set[int]) -> list[Candidate]:
    """Motion that jumps far above its neighbours while the picture stays the same shot."""
    flow = np.array([0.0 if math.isnan(f) else f for f in scan.flow_median])
    candidates = []
    for i in range(2, scan.frame_count - 2):
        if i in cut_frames:
            continue
        neighbours = np.concatenate([flow[max(1, i - 4) : i], flow[i + 1 : i + 5]])
        neighbours = neighbours[neighbours > t.freeze_flow_px]  # ignore still pairs
        local = float(np.median(neighbours)) if neighbours.size else 0.0
        if local < t.skip_min_frame_motion:  # a still or locked-off frame cannot show a skip
            continue
        ratio = flow[i] / max(local, 1e-3)
        if ratio >= t.skip_candidate_ratio and scan.hist_distance[i] < t.shot_stable_hist_distance:
            candidates.append(
                Candidate(
                    SKIP,
                    i,
                    i,
                    ratio / t.skip_candidate_ratio,
                    {
                        "median_flow_px": float(flow[i]),
                        "local_median_flow_px": local,
                        "flow_ratio": float(ratio),
                    },
                )
            )
    return candidates


def find_candidates(scan: ScanResult, t: Thresholds) -> list[Candidate]:
    black = find_black(scan, t)
    black_frames = {i for c in black for i in range(c.start, c.end + 1)}
    cuts = find_shot_changes(scan, t)
    cut_frames = {c.start for c in cuts}
    # A cut into or out of a black run is explained by the black run itself.
    cuts = [c for c in cuts if c.start not in black_frames and c.start - 1 not in black_frames]
    candidates = (
        black
        + find_freezes(scan, t, black_frames)
        + cuts
        + find_damage(scan, t)
        + find_skips(scan, t, cut_frames)
    )
    return sorted(candidates, key=lambda c: (c.start, c.kind))
