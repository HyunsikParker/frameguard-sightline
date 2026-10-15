"""Limits and thresholds for FrameGuard Sightline.

Every number that changes a decision lives here so that a reviewer can see, in one place,
what the analysis treats as black, frozen, cut or damaged, and how much work the agent
may do before it must stop and ask a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InputLimits:
    """Hard bounds on what a job may decode. Anything larger is rejected, not truncated."""

    max_input_bytes: int = 200 * 1024 * 1024
    max_frames: int = 3000
    max_width: int = 3840
    max_height: int = 2160
    # Frames are measured at this width; thumbnails are written at thumb_width.
    analysis_width: int = 320
    thumb_width: int = 192


@dataclass(frozen=True)
class Thresholds:
    """Detector thresholds. Values are on 8-bit luma unless noted."""

    # Black or near-empty frames: dark and nearly featureless.
    black_luma_mean: float = 18.0
    # Grain or sensor noise on a black frame raises entropy but keeps the spread small.
    black_luma_std: float = 6.0
    # A fade passes through dark frames gradually; a defect drops abruptly.
    black_abrupt_luma_drop: float = 25.0

    # Frozen runs: near-identical perceptual hash and almost no optical flow.
    freeze_hash_distance: int = 2  # out of 64 bits
    freeze_flow_px: float = 0.05  # trimmed mean Farneback magnitude at analysis width
    # A pair is still when almost no pixel changes beyond codec noise (flow is unreliable here).
    freeze_changed_fraction: float = 0.0005
    # A shot is moving when at least this fraction of pixels changes between frames.
    moving_changed_fraction: float = 0.004
    # Full-resolution confirmation: a stall's typical pair changes almost no pixels.
    freeze_confirm_changed: float = 0.0001
    freeze_min_frames: int = 6
    # Follow-up check at higher resolution: largest mean absolute pixel change allowed.
    freeze_confirm_mad: float = 0.6

    # Hard cuts: colour distribution and edge layout both change at once.
    cut_hist_distance: float = 0.42  # Bhattacharyya distance between HSV histograms
    cut_edge_change: float = 0.35  # relative change in edge density
    # A skip scales the shot's own motion; a transient event (flash, explosion) does not.
    skip_min_field_agreement: float = 0.6
    skip_candidate_ratio: float = 2.6
    skip_confirm_ratio: float = 2.4
    # Skips are judged on whole-frame (median) motion so a fast local subject is not a skip.
    skip_min_frame_motion: float = 0.15
    # A cut that returns to the previous shot within this many frames is a flash frame.
    flash_max_frames: int = 2
    # Shots on either side of an intended cut must each be internally stable.
    shot_stability_window: int = 6
    shot_stable_hist_distance: float = 0.25
    # Two frames belong to the same shot only if their layouts also match (title cards share colours).
    same_shot_hash_distance: int = 16
    # Black runs this short between two different shots may be a timeline gap, so a person decides.
    short_black_frames: int = 3

    # Damaged frames: 8x8 block-boundary energy far above the clip's own baseline.
    damage_block_ratio: float = 1.9
    damage_zscore: float = 6.0
    # Flat frames (black, white, plain titles) make the ratio explode on faint compression steps.
    damage_min_step: float = 4.0
    # Follow-up: a 16x16 block is an outlier when it changes by this much against both neighbours.
    damage_block_change: float = 25.0
    damage_outlier_fraction: float = 0.01


@dataclass(frozen=True)
class AgentBudget:
    """How much follow-up work the agent may do on one job."""

    max_tool_calls: int = 60
    max_redecoded_frames: int = 1200
    # Findings whose confidence stays inside this band after follow-up go to a person.
    review_band: tuple[float, float] = (0.35, 0.75)


@dataclass(frozen=True)
class Settings:
    limits: InputLimits = field(default_factory=InputLimits)
    thresholds: Thresholds = field(default_factory=Thresholds)
    budget: AgentBudget = field(default_factory=AgentBudget)


DEFAULT_SETTINGS = Settings()
