"""Tests for measurements, detectors, the agent loop, limits and the approval/action flow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from sightline.actions import ApprovalError, apply_approved, record_decision
from sightline.config import DEFAULT_SETTINGS, AgentBudget, InputLimits, Settings
from sightline.io import InputRejected, probe
from sightline.measures import block_boundary_ratio, hamming, perceptual_hash
from sightline.pipeline import analyze
from sightline.report import write_report
from sightline.synth import MovingScene, corrupt_blocks, title_card, write_clip


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _write_sequence(directory: Path, frames: list[np.ndarray]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(directory / f"{i:05d}.png"), frame)
    return directory


def _flagged(job, kind):
    return [f for f in job.report.findings if f.kind == kind and f.verdict != "dismissed"]


# ----------------------------------------------------------------------------- measures
def test_perceptual_hash_is_stable_and_discriminative():
    a = MovingScene(1).next()
    b = MovingScene(2).next()
    assert hamming(perceptual_hash(gray(a)), perceptual_hash(gray(a.copy()))) == 0
    assert hamming(perceptual_hash(gray(a)), perceptual_hash(gray(b))) > 10


def test_block_ratio_rises_on_macroblock_damage():
    frame = MovingScene(3).next()
    damaged = corrupt_blocks(frame, np.random.default_rng(0), fraction=0.15)
    assert block_boundary_ratio(gray(damaged)) > 1.5 * block_boundary_ratio(gray(frame))


# ---------------------------------------------------------------------------- detectors
def test_abrupt_black_is_a_defect_but_fade_is_dismissed(tmp_path):
    scene = MovingScene(4)
    frames = [scene.next() for _ in range(30)]
    frames += [np.zeros_like(frames[0])] * 2  # abrupt black frames
    frames += [scene.next() for _ in range(30)]
    last = frames[-1]
    frames += [(last.astype(np.float32) * (1 - k / 12)).astype(np.uint8) for k in range(1, 13)]  # fade out
    job = analyze(str(_write_sequence(tmp_path / "clip", frames)))
    black = _flagged(job, "black_frame")
    assert [(f.start, f.end) for f in black] == [(30, 31)]
    assert any(f.kind == "fade" and f.verdict == "dismissed" for f in job.report.findings)


def test_freeze_inside_motion_is_a_defect_but_title_hold_and_slow_motion_are_not(tmp_path):
    scene = MovingScene(5)
    frames = [scene.next() for _ in range(30)]
    held = scene.next()
    frames += [held.copy() for _ in range(10)]  # stall inside a moving shot
    frames += [scene.next() for _ in range(25)]
    frames += [title_card("Title", 5)] * 24  # intended hold after a cut
    slow = MovingScene(6, speed=0.08)
    frames += [slow.next() for _ in range(30)]  # slow but real motion
    job = analyze(str(_write_sequence(tmp_path / "clip", frames)))
    freezes = _flagged(job, "frozen_run")
    assert len(freezes) == 1 and freezes[0].start in (29, 30, 31)
    kinds = {f.kind for f in job.report.findings if f.verdict == "dismissed"}
    assert "held_shot" in kinds


def test_flash_frame_is_paired_into_one_finding(tmp_path):
    scene, intruder = MovingScene(7), MovingScene(8)
    frames = [scene.next() for _ in range(40)]
    frames.append(intruder.next())
    scene.next()
    frames += [scene.next() for _ in range(40)]
    job = analyze(str(_write_sequence(tmp_path / "clip", frames)))
    flashes = _flagged(job, "flash_frame")
    assert [(f.start, f.end) for f in flashes] == [(40, 40)]
    # The cut back into the shot must not be reported separately.
    assert not _flagged(job, "shot_change")


def test_motion_spike_next_to_a_stall_is_not_a_skip(tmp_path):
    scene = MovingScene(9)
    frames = [scene.next() for _ in range(30)]
    held = scene.next()
    frames += [held.copy() for _ in range(10)]
    frames += [scene.next() for _ in range(30)]
    job = analyze(str(_write_sequence(tmp_path / "clip", frames)))
    assert not _flagged(job, "frame_skip")


# ------------------------------------------------------------------------- whole clips
def test_seeded_clip_defects_are_found(tmp_path):
    clip = write_clip(tmp_path, 4)
    labels = json.loads((tmp_path / "clip_0004.labels.json").read_text())["labels"]
    job = analyze(str(clip))
    flagged = [f for f in job.report.findings if f.verdict != "dismissed"]
    for label in labels:
        assert any(
            f.kind == label["kind"] and f.start <= label["end"] + 1 and label["start"] <= f.end + 1
            for f in flagged
        ), label


def test_every_check_is_traced_with_its_trigger(tmp_path):
    job = analyze(str(write_clip(tmp_path, 5)))
    assert job.report.trace
    for step in job.report.trace:
        assert step.tool and step.trigger and step.decision
    for finding in job.report.findings:
        assert finding.steps, finding


def test_budget_exhaustion_sends_unchecked_candidates_to_review(tmp_path):
    clip = write_clip(tmp_path, 5)
    tight = Settings(budget=AgentBudget(max_tool_calls=2))
    job = analyze(str(clip), tight)
    assert job.report.budget_exhausted
    assert any(f.verdict == "needs_review" and "budget" in f.reason for f in job.report.findings)


# ------------------------------------------------------------------------------- limits
def test_oversized_frames_are_rejected(tmp_path):
    big = np.zeros((300, 4000, 3), dtype=np.uint8)
    seq = _write_sequence(tmp_path / "big", [big])
    with pytest.raises(InputRejected):
        probe(seq, InputLimits())


def test_too_many_frames_is_rejected(tmp_path):
    seq = _write_sequence(tmp_path / "long", [MovingScene(1).next()] * 5)
    with pytest.raises(InputRejected):
        probe(seq, replace(InputLimits(), max_frames=4))


def test_unreadable_video_is_rejected(tmp_path):
    bogus = tmp_path / "broken.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(InputRejected):
        probe(bogus, DEFAULT_SETTINGS.limits)


# ---------------------------------------------------------------- approval and actions
def test_actions_need_approval_and_repairs_are_verified(tmp_path):
    scene, intruder = MovingScene(10), MovingScene(11)
    frames = [scene.next() for _ in range(30)]
    frames.append(intruder.next())
    scene.next()
    frames += [scene.next() for _ in range(30)]
    clip = _write_sequence(tmp_path / "clip", frames)
    job_dir = tmp_path / "job"
    write_report(analyze(str(clip)), job_dir)

    # Nothing happens before a person decides.
    untouched = apply_approved(job_dir, str(clip))
    assert untouched["preview"] is None

    record_decision(job_dir, "flash_frame:30-30", "approve", by="tester")
    with pytest.raises(ApprovalError):
        record_decision(job_dir, "flash_frame:30-30", "approve", by="tester")
    result = apply_approved(job_dir, str(clip))
    assert result["repaired_ranges"] == ["flash_frame:30-30"]
    assert result["verification"]["clean"] is True
    log = json.loads((job_dir / "actions.json").read_text())
    assert [e["event"] for e in log][-3:] == ["decision", "actions_applied", "repairs_verified"]


# ------------------------------------------------------------------- local review server
def test_local_server_serves_byte_ranges_and_refuses_traversal(tmp_path):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from sightline.web import Store, make_handler

    store = Store(tmp_path)
    job_id = "c" * 32
    store.jobs[job_id] = {"job_id": job_id, "status": "done", "input_name": "clip.mp4"}
    media = store.job_dir(job_id) / "report"
    media.mkdir(parents=True)
    (media / "repair_preview.mp4").write_bytes(bytes(range(256)) * 4)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/local-media/{job_id}"
    try:
        request = urllib.request.Request(f"{base}/repair_preview.mp4", headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers["content-range"] == "bytes 10-19/1024"
            assert response.read() == bytes(range(10, 20))
        with urllib.request.urlopen(f"{base}/repair_preview.mp4") as response:
            assert response.status == 200 and response.headers["accept-ranges"] == "bytes"
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"{base}/../../etc/passwd")
        assert refused.value.code == 404
    finally:
        server.shutdown()
