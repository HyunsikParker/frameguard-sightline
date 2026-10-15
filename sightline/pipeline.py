"""One analysis job: probe, first-pass scan, agent loop, and the job record."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict, dataclass

import cv2

from .agent import AgentReport, SightlineAgent
from .config import DEFAULT_SETTINGS, Settings
from .io import SourceInfo, probe
from .scan import ScanResult, scan


@dataclass
class JobResult:
    info: SourceInfo
    scan: ScanResult
    report: AgentReport
    timings: dict[str, float]
    peak_memory_mb: float
    settings: Settings = DEFAULT_SETTINGS

    def summary(self) -> dict:
        findings = self.report.findings
        return {
            "source": asdict(self.info),
            "opencv_version": cv2.__version__,
            "frames_scanned": self.scan.frame_count,
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "ms_per_frame_first_pass": round(1000 * self.timings["scan"] / max(1, self.scan.frame_count), 2),
            "peak_python_memory_mb": round(self.peak_memory_mb, 1),
            "tool_calls": self.report.tool_calls,
            "redecoded_frames": self.report.redecoded_frames,
            "budget_exhausted": self.report.budget_exhausted,
            "defects": sum(f.verdict == "defect" for f in findings),
            "needs_review": sum(f.verdict == "needs_review" for f in findings),
            "dismissed": sum(f.verdict == "dismissed" for f in findings),
        }


def analyze(path: str, settings: Settings = DEFAULT_SETTINGS) -> JobResult:
    tracemalloc.start()
    t0 = time.perf_counter()
    info = probe(path, settings.limits)
    t1 = time.perf_counter()
    first_pass = scan(info, settings.limits)
    t2 = time.perf_counter()
    report = SightlineAgent(info, first_pass, settings).run()
    t3 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings = {"probe": t1 - t0, "scan": t2 - t1, "agent": t3 - t2, "total": t3 - t0}
    return JobResult(info, first_pass, report, timings, peak / 1e6, settings)
