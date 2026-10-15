"""Perception-decision-action loop.

The agent takes the first-pass candidates and, for each one, chooses a follow-up check
based on what OpenCV measured. A check re-decodes a short window at higher resolution,
measures again, and returns a verdict:

* ``defect``        the evidence holds; an action is proposed for a person to approve
* ``dismissed``     the event is intended material (a cut, a fade, a held title, slow motion)
* ``needs_review``  the evidence is ambiguous or the budget ran out; a person decides

Every check is written to the trace with the measurement that triggered it, the numbers it
produced and the decision taken, so a reviewer can replay why a frame was flagged.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .config import Settings
from .detectors import BLACK, CUT, DAMAGE, FREEZE, SKIP, Candidate, find_candidates
from .io import SourceInfo, iter_frames, resize_to_width
from .measures import (
    field_agreement,
    flow_field,
    hamming,
    histogram_distance,
    hsv_histogram,
    median_flow_magnitude,
    perceptual_hash,
)
from .scan import ScanResult

FLASH = "flash_frame"
DEFECT, DISMISSED, REVIEW = "defect", "dismissed", "needs_review"
CONFIRM_WIDTH = 960  # follow-up checks look at up to this width

# What a person is asked to approve for each confirmed defect kind.
PROPOSED_ACTIONS = {
    BLACK: "Replace the black frames with the neighbouring good frame in a repair preview, then re-export.",
    FLASH: "Replace the inserted frames with the neighbouring frame of the same shot in a repair preview.",
    FREEZE: "Re-render the frozen range from the source project; a hold cannot be repaired from the export.",
    DAMAGE: "Re-export the damaged range; show the block-damage heatmap to the editor first.",
    SKIP: "Re-render around the jump; dropped frames cannot be recovered from the export.",
}
REPAIRABLE = {BLACK, FLASH, DAMAGE}


@dataclass
class TraceStep:
    step: int
    tool: str
    target: dict
    trigger: str
    result: dict
    decision: str


@dataclass
class Finding:
    kind: str
    start: int
    end: int
    verdict: str
    confidence: float
    reason: str
    evidence: dict = field(default_factory=dict)
    steps: list[int] = field(default_factory=list)
    proposed_action: str | None = None
    approval: str = "not_required"  # "pending" for defects and reviews until a person answers


@dataclass
class AgentReport:
    findings: list[Finding]
    trace: list[TraceStep]
    tool_calls: int
    redecoded_frames: int
    budget_exhausted: bool

    def to_dict(self) -> dict:
        return {
            "findings": [asdict(f) for f in self.findings],
            "trace": [asdict(s) for s in self.trace],
            "tool_calls": self.tool_calls,
            "redecoded_frames": self.redecoded_frames,
            "budget_exhausted": self.budget_exhausted,
        }


class SightlineAgent:
    def __init__(self, info: SourceInfo, scan: ScanResult, settings: Settings):
        self.info = info
        self.scan = scan
        self.settings = settings
        self.t = settings.thresholds
        self.trace: list[TraceStep] = []
        self.findings: list[Finding] = []
        self.tool_calls = 0
        self.redecoded = 0
        self.budget_exhausted = False
        # Frames covered by first-pass candidates; follow-up checks avoid them as reference frames.
        self.suspect_frames: set[int] = set()

    # ----------------------------------------------------------------- tool plumbing
    def _frames(self, start: int, stop: int) -> list[np.ndarray]:
        """Re-decode frames [start, stop) at confirmation resolution, counting the cost."""
        start, stop = max(0, start), min(self.info.frame_count, stop)
        frames = [
            resize_to_width(f.bgr, CONFIRM_WIDTH)
            for f in iter_frames(self.info, self.settings.limits, start, stop)
        ]
        self.redecoded += len(frames)
        return frames

    def _budget_left(self, frames_needed: int) -> bool:
        b = self.settings.budget
        return self.tool_calls < b.max_tool_calls and self.redecoded + frames_needed <= b.max_redecoded_frames

    def _record(self, tool: str, candidate: Candidate, trigger: str, result: dict, decision: str) -> int:
        self.tool_calls += 1
        step = TraceStep(
            len(self.trace) + 1,
            tool,
            {"kind": candidate.kind, "start": candidate.start, "end": candidate.end},
            trigger,
            result,
            decision,
        )
        self.trace.append(step)
        return step.step

    def _add(
        self,
        kind: str,
        c: Candidate,
        verdict: str,
        confidence: float,
        reason: str,
        evidence: dict,
        steps: list[int],
        start: int | None = None,
        end: int | None = None,
    ) -> Finding:
        low, high = self.settings.budget.review_band
        if verdict == DEFECT and low <= confidence < high:
            verdict, reason = REVIEW, reason + " Confidence is inside the review band, so a person decides."
        finding = Finding(
            kind,
            c.start if start is None else start,
            c.end if end is None else end,
            verdict,
            round(confidence, 3),
            reason,
            evidence,
            steps,
        )
        if verdict == DEFECT:
            finding.proposed_action = PROPOSED_ACTIONS.get(kind)
            finding.approval = "pending"
        elif verdict == REVIEW:
            finding.proposed_action = "Decide whether this is a defect; the agent could not settle it."
            finding.approval = "pending"
        self.findings.append(finding)
        return finding

    # ------------------------------------------------------------- shared measurements
    def _gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> dict:
        """Colour-distribution and layout distance between two frames."""
        return {
            "hist_distance": round(histogram_distance(hsv_histogram(a), hsv_histogram(b)), 3),
            "hash_distance": hamming(perceptual_hash(self._gray(a)), perceptual_hash(self._gray(b))),
        }

    def _is_same_shot(self, similarity: dict) -> bool:
        """Same shot when both colours and layout match; colour alone is fooled by title cards."""
        return (
            similarity["hist_distance"] < self.t.shot_stable_hist_distance
            and similarity["hash_distance"] <= self.t.same_shot_hash_distance
        )

    def _frame_pair(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray] | None:
        if a < 0 or b >= self.info.frame_count:
            return None
        first = self._frames(a, a + 1)
        second = self._frames(b, b + 1)
        return (first[0], second[0]) if first and second else None

    # ------------------------------------------------------------------- the checks
    def check_black(self, c: Candidate) -> None:
        """Black inside a continuing shot is a defect; fades, leaders and cuts to black are not."""
        last = self.info.frame_count - 1
        lo = max(0, c.start - 8)
        lumas = [s.luma_mean for s in self.scan.stats[lo : c.start + 1]]
        steps_down = [a - b for a, b in zip(lumas, lumas[1:], strict=False)]
        gradual = (
            len(steps_down) >= 4
            and all(d > -2.0 for d in steps_down[-4:])
            and max(steps_down[-4:], default=0) < self.t.black_abrupt_luma_drop
        )
        result: dict = {"luma_before": [round(v, 1) for v in lumas], "frames": c.length}
        trigger = f"luma {c.measurements['darkest_luma']:.1f} below {self.t.black_luma_mean}"

        if gradual:
            step = self._record("check_black", c, trigger, result, "dismissed: fade")
            reason = "Luma declines gradually into the dark run, which is a fade."
            self._add("fade", c, DISMISSED, 0.9, reason, result, [step])
            return
        if c.start == 0 or c.end == last:
            step = self._record("check_black", c, trigger, result, "dismissed: head or tail black")
            self._add(
                "leader_black",
                c,
                DISMISSED,
                0.85,
                "Dark run at the start or end of the clip.",
                result,
                [step],
            )
            return

        pair = self._frame_pair(c.start - 1, c.end + 1)
        similarity = self._similarity(*pair) if pair else {}
        result["before_vs_after"] = similarity
        if pair and self._is_same_shot(similarity):
            step = self._record("check_black", c, trigger, result, "defect: black inside one shot")
            reason = "The same shot continues on both sides of the black frames."
            self._add(BLACK, c, DEFECT, 0.95, reason, result, [step])
        elif c.length <= self.t.short_black_frames:
            step = self._record("check_black", c, trigger, result, "needs review: short black at an edit")
            reason = "A few black frames between two different shots: a timeline gap or a quick dip."
            self._add(BLACK, c, REVIEW, 0.5, reason, result, [step])
        else:
            step = self._record("check_black", c, trigger, result, "dismissed: cut to black")
            reason = "A held black between two different shots, which is an editorial cut to black."
            self._add("cut_to_black", c, DISMISSED, 0.8, reason, result, [step])

    def check_shot_change(self, c: Candidate, pending: list[Candidate]) -> None:
        """A shot that returns within a couple of frames hides a flash; otherwise this is an edit."""
        k = self.t.flash_max_frames
        window = self._frames(c.start - 1, c.start + k + 2)
        compared = [self._similarity(window[0], frame) for frame in window[1:]]
        result: dict = {"vs_previous_shot": compared}
        trigger = f"histogram jump {c.measurements['hist_distance']:.2f}"
        returns = [i for i, s in enumerate(compared) if 1 <= i <= k and self._is_same_shot(s)]
        if returns:
            length = returns[0]
            start, end = c.start, c.start + length - 1
            frames = "1 frame" if length == 1 else f"{length} frames"
            step = self._record("check_shot_change", c, trigger, result, f"defect: flash of {frames}")
            # The matching cut back into the shot is part of this flash; drop it from the queue.
            pending[:] = [p for p in pending if not (p.kind == CUT and start < p.start <= end + 1)]
            what = "One foreign frame" if length == 1 else f"{length} foreign frames"
            reason = f"{what} inside one shot; the shot resumes after them."
            self._add(FLASH, c, DEFECT, 0.9 if length == 1 else 0.85, reason, result, [step], start, end)
            return
        stable = self._shot_stable(c.start - 1, -1) and self._shot_stable(c.start, +1)
        result["both_sides_stable"] = stable
        note = "Both shots are internally stable." if stable else "A busy shot on one side; still an edit."
        step = self._record("check_shot_change", c, trigger, result, "dismissed: edit point")
        self._add(
            "intended_cut",
            c,
            DISMISSED,
            0.9 if stable else 0.7,
            f"The picture does not return. {note}",
            result,
            [step],
        )

    def _shot_stable(self, index: int, direction: int) -> bool:
        w = self.t.shot_stability_window
        distances = []
        for i in range(index + direction, index + direction * (w + 1), direction):
            j = i if direction > 0 else i + 1
            if 1 <= j < self.scan.frame_count and not math.isnan(self.scan.hist_distance[j]):
                distances.append(self.scan.hist_distance[j])
        return bool(distances) and max(distances) < self.t.shot_stable_hist_distance

    def check_freeze(self, c: Candidate) -> None:
        """Identical frames that interrupt a shot which resumes are a stall; holds and slow motion are not."""
        frames = self._frames(c.start, c.end + 1)
        greys = [self._gray(f) for f in frames]
        changed = [float((cv2.absdiff(a, b) > 6).mean()) for a, b in zip(greys, greys[1:], strict=False)]
        # The pair entering a hold carries a codec transition, so the typical (median) pair decides.
        max_changed = max(changed) if changed else 0.0
        median_changed = float(np.median(changed)) if changed else 0.0
        starts_at_cut = c.start >= 1 and self.scan.hist_distance[c.start] >= self.t.cut_hist_distance * 0.8
        changes = self.scan.changed_fraction
        before = [v for v in changes[max(1, c.start - 6) : c.start] if not math.isnan(v)]
        after = [v for v in changes[c.end + 2 : c.end + 8] if not math.isnan(v)]
        moving_before = bool(before) and float(np.median(before)) > self.t.moving_changed_fraction
        moving_after = bool(after) and float(np.median(after)) > self.t.moving_changed_fraction
        pair = self._frame_pair(c.end, c.end + 1)
        # After a stall the picture jumps ahead, so the layout may differ; the colours must continue,
        # and either the layout matches or the shot keeps moving (consecutive title cards do neither).
        after_hold = self._similarity(*pair) if pair else None
        resumes = (
            bool(after_hold)
            and after_hold["hist_distance"] < self.t.shot_stable_hist_distance
            and (after_hold["hash_distance"] <= self.t.same_shot_hash_distance or moving_after)
        )
        result = {
            "max_changed_pixel_fraction": round(max_changed, 5),
            "median_changed_pixel_fraction": round(median_changed, 6),
            "starts_at_shot_change": bool(starts_at_cut),
            "moving_before": moving_before,
            "same_shot_resumes": resumes,
            "moving_after": moving_after,
        }
        trigger = f"flow {c.measurements['max_flow_px']:.3f}px over {c.length} frames"
        if median_changed >= self.t.freeze_confirm_changed:
            step = self._record("check_freeze", c, trigger, result, "dismissed: slow motion")
            reason = "Pixels keep changing at full resolution; the motion is slow, not stopped."
            self._add("slow_motion", c, DISMISSED, 0.85, reason, result, [step])
        elif starts_at_cut or not resumes:
            # A still that begins at an edit, or one that ends in an edit or the clip end, is a hold.
            step = self._record("check_freeze", c, trigger, result, "dismissed: held shot")
            reason = "A still shot bounded by an edit point or the clip end, such as a title card."
            self._add("held_shot", c, DISMISSED, 0.8, reason, result, [step])
        elif moving_before or moving_after:
            step = self._record("check_freeze", c, trigger, result, "defect: stall inside a moving shot")
            reason = f"{c.length} identical frames interrupt a shot that moves around them."
            self._add(FREEZE, c, DEFECT, 0.92, reason, result, [step])
        else:
            step = self._record("check_freeze", c, trigger, result, "needs review")
            reason = "Frames are identical inside one shot, but the shot barely moves around them."
            self._add(FREEZE, c, REVIEW, 0.55, reason, result, [step])

    def _outlier_blocks(self, previous: np.ndarray, current: np.ndarray, following: np.ndarray) -> float:
        """Fraction of 16x16 blocks that differ from both neighbours while the neighbours agree."""
        size = 16
        h, w = current.shape[:2]
        h, w = h - h % size, w - w % size

        def block_means(x: np.ndarray) -> np.ndarray:
            return x[:h, :w].reshape(h // size, size, w // size, size).mean(axis=(1, 3))

        vs_previous, vs_following, neighbours = (
            block_means(cv2.absdiff(a, b).astype(np.float32))
            for a, b in ((current, previous), (current, following), (previous, following))
        )
        change = np.minimum(vs_previous, vs_following)
        outliers = (change > self.t.damage_block_change) & (neighbours < 0.5 * change)
        return float(outliers.mean())

    def _clean_neighbour(self, index: int, direction: int, reach: int = 6) -> int | None:
        """Nearest frame in the given direction that no other candidate covers."""
        for step in range(1, reach + 1):
            j = index + direction * step
            if not 0 <= j < self.info.frame_count:
                return None
            if j not in self.suspect_frames:
                return j
        return None

    def check_damage(self, c: Candidate) -> None:
        """Damage sits on the block grid and appears only in these frames, not in their neighbours."""
        previous = self._clean_neighbour(c.start, -1)
        following = self._clean_neighbour(c.end, +1)
        fractions: list[float] = []
        if previous is not None and following is not None:
            pair = self._frame_pair(previous, following)
            if pair:
                before, after = (self._gray(f) for f in pair)
                frames = self._frames(c.start, c.end + 1)
                fractions = [self._outlier_blocks(before, self._gray(f), after) for f in frames]
        peak = max(fractions, default=0.0)
        strong = c.measurements["peak_block_ratio"] >= 2 * self.t.damage_block_ratio
        result = {
            "reference_frames": [previous, following],
            "outlier_block_fraction_by_frame": [round(f, 4) for f in fractions],
            "peak_block_ratio": round(c.measurements["peak_block_ratio"], 3),
            "clip_median_ratio": round(c.measurements["clip_median_ratio"], 3),
        }
        trigger = (
            f"block ratio {c.measurements['peak_block_ratio']:.2f} "
            f"vs clip median {c.measurements['clip_median_ratio']:.2f}"
        )
        if peak >= self.t.damage_outlier_fraction:
            step = self._record("check_damage", c, trigger, result, "defect: temporal outlier blocks")
            reason = "Blocks change in these frames only, while the frames around them agree."
            self._add(DAMAGE, c, DEFECT, 0.9, reason, result, [step])
        elif strong:
            step = self._record("check_damage", c, trigger, result, "needs review: strong grid, no outliers")
            reason = "Very strong block-grid edges that the neighbour comparison could not confirm."
            self._add(DAMAGE, c, REVIEW, 0.5, reason, result, [step])
        else:
            step = self._record("check_damage", c, trigger, result, "dismissed: consistent over time")
            reason = "The sharp block edges persist across neighbouring frames (graphics or texture)."
            self._add("texture", c, DISMISSED, 0.75, reason, result, [step])

    def check_skip(self, c: Candidate) -> None:
        """A motion spike explained by damage nearby is not a skip; otherwise confirm with denser flow."""
        for f in self.findings:
            if (
                f.kind in (DAMAGE, FREEZE, BLACK, FLASH)
                and f.verdict == DEFECT
                and f.start - 2 <= c.start <= f.end + 2
            ):
                evidence = {"neighbour_defect": f.kind, "range": [f.start, f.end]}
                step = self._record(
                    "check_skip",
                    c,
                    f"motion spike next to confirmed {f.kind}",
                    evidence,
                    "dismissed: explained by neighbour",
                )
                self._add(
                    "neighbour_side_effect",
                    c,
                    DISMISSED,
                    0.8,
                    f"The spike comes from the adjacent {f.kind.replace('_', ' ')}, not from dropped frames.",
                    evidence,
                    [step],
                )
                return
        # Dense flow at confirmation width misreads smooth regions as still, so the motion test runs at
        # the first-pass width; this check adds the neighbour explanation and the direction test.
        width = self.settings.limits.analysis_width
        frames = [resize_to_width(f, width) for f in self._frames(c.start - 3, c.start + 3)]
        greys = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        fields = [flow_field(a, b) for a, b in zip(greys, greys[1:], strict=False)]
        flows = [median_flow_magnitude(f) for f in fields]
        centre = min(2, len(flows) - 1)
        # Still pairs (a stall or a hold) say nothing about the shot's normal pace.
        others = [i for i, v in enumerate(flows) if i != centre and v > self.t.freeze_flow_px]
        ratio = flows[centre] / max(float(np.median([flows[i] for i in others])), 1e-3) if others else 0.0
        reference = sum(fields[i] for i in others) if others else None
        agreement = field_agreement(fields[centre], reference) if reference is not None else 0.0
        result = {
            "flow_px_window": [round(v, 2) for v in flows],
            "spike_ratio": round(ratio, 2),
            "direction_agreement": round(agreement, 3),
        }
        if ratio >= self.t.skip_confirm_ratio and agreement < self.t.skip_min_field_agreement:
            # Dense flow is unreliable for large jumps, so a weak direction match is not proof either way.
            step = self._record(
                "check_skip",
                c,
                "motion spike whose direction only partly matches the shot",
                result,
                "needs review",
            )
            reason = "A motion jump that may be dropped frames or a fast transient; a person should look."
            self._add(SKIP, c, REVIEW, 0.5, reason, result, [step])
            return
        if ratio >= self.t.skip_confirm_ratio:
            step = self._record(
                "check_skip",
                c,
                f"median flow {c.measurements['median_flow_px']:.2f}px "
                f"vs local {c.measurements['local_median_flow_px']:.2f}px",
                result,
                "defect: jump",
            )
            self._add(
                SKIP,
                c,
                DEFECT,
                min(0.95, 0.6 + 0.08 * ratio),
                "Motion jumps far beyond the shot's own pace, as if frames were dropped.",
                result,
                [step],
            )
        else:
            step = self._record(
                "check_skip", c, "motion spike did not hold at higher resolution", result, "dismissed"
            )
            self._add(
                "motion_variation",
                c,
                DISMISSED,
                0.6,
                "Denser flow shows ordinary motion variation.",
                result,
                [step],
            )

    # --------------------------------------------------------------------- the loop
    def run(self) -> AgentReport:
        candidates = find_candidates(self.scan, self.t)
        for candidate in candidates:
            # Cuts and holds are still valid pictures of their shots; the rest cannot serve as references.
            if candidate.kind in (BLACK, DAMAGE, SKIP):
                self.suspect_frames.update(range(candidate.start, candidate.end + 1))
        # Damage first so that motion spikes it causes can be explained away later.
        order = {DAMAGE: 0, BLACK: 1, CUT: 2, FREEZE: 3, SKIP: 4}
        pending = sorted(candidates, key=lambda c: (order[c.kind], c.start))
        checks: dict[str, Callable[[Candidate], None]] = {
            BLACK: self.check_black,
            FREEZE: self.check_freeze,
            DAMAGE: self.check_damage,
            SKIP: self.check_skip,
        }
        while pending:
            c = pending.pop(0)
            if not self._budget_left(c.length + 8):
                self.budget_exhausted = True
                step = self._record(
                    "budget_stop",
                    c,
                    "tool or decode budget reached",
                    {"tool_calls": self.tool_calls, "redecoded_frames": self.redecoded},
                    "needs review",
                )
                self._add(
                    c.kind,
                    c,
                    REVIEW,
                    0.5,
                    "Not checked: the job's follow-up budget ran out.",
                    dict(c.measurements),
                    [step],
                )
                continue
            if c.kind == CUT:
                self.check_shot_change(c, pending)
            else:
                checks[c.kind](c)
        self.findings.sort(key=lambda f: (f.start, f.kind))
        return AgentReport(self.findings, self.trace, self.tool_calls, self.redecoded, self.budget_exhausted)
