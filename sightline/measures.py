"""Per-frame and frame-pair measurements computed with OpenCV 5.

The detectors and the agent only ever reason over these numbers, so each one is cheap,
deterministic and stored with the finding that used it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HIST_BINS = (16, 16)  # hue, saturation


@dataclass
class FrameStats:
    index: int
    luma_mean: float
    luma_std: float
    entropy_bits: float
    phash: int
    edge_density: float
    block_ratio: float
    block_step: float  # mean absolute step across block boundaries, in 8-bit levels
    hist: np.ndarray  # normalised HSV histogram, float32


def luma_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the 8-bit luma histogram, in bits (0..8)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-(p * np.log2(p)).sum())


def perceptual_hash(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash: low-frequency DCT terms compared with their median."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeffs = cv2.dct(small)[:8, :8].ravel()
    ac = coeffs[1:]  # drop the DC term, which only tracks overall brightness
    bits = ac > np.median(ac)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def hsv_histogram(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance: 0 for identical colour distributions, 1 for disjoint."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 140)
    return float(np.count_nonzero(edges)) / edges.size


def block_boundary_stats(gray: np.ndarray, block: int = 8) -> tuple[float, float]:
    """Block-grid step and its ratio to the step inside blocks.

    Returns (ratio, boundary_step). Clean footage has a ratio close to 1.0. Corrupted exports
    show strong discontinuities on the block grid. On flat frames (black, white, plain titles)
    the ratio is unstable because the inner step is near zero, so callers also require the
    boundary step itself to be large before calling a frame damaged.
    """
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    col = np.arange(dx.shape[1])
    row = np.arange(dy.shape[0])
    on_x = (col % block) == block - 1
    on_y = (row % block) == block - 1
    boundary = np.concatenate([dx[:, on_x].ravel(), dy[on_y, :].ravel()])
    inner = np.concatenate([dx[:, ~on_x].ravel(), dy[~on_y, :].ravel()])
    if boundary.size == 0 or inner.size == 0:
        return 1.0, 0.0
    step = float(boundary.mean())
    return (step + 1e-3) / (float(inner.mean()) + 1e-3), step


def block_boundary_ratio(gray: np.ndarray, block: int = 8) -> float:
    return block_boundary_stats(gray, block)[0]


def frame_stats(index: int, bgr: np.ndarray) -> FrameStats:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    block_ratio, block_step = block_boundary_stats(gray)
    return FrameStats(
        index=index,
        luma_mean=float(mean[0][0]),
        luma_std=float(std[0][0]),
        entropy_bits=luma_entropy(gray),
        phash=perceptual_hash(gray),
        edge_density=edge_density(gray),
        block_ratio=block_ratio,
        block_step=block_step,
        hist=hsv_histogram(bgr),
    )


def flow_field(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Dense Farneback optical flow (H x W x 2, pixels) between two grey frames."""
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


# Farneback is numerically unstable on flat or saturated regions: on two identical real frames
# about 1% of pixels can report displacements of hundreds of pixels. Statistics therefore drop
# the largest 2% of magnitudes before averaging.
FLOW_TRIM_PERCENTILE = 98.0


def trimmed_flow(flow: np.ndarray) -> np.ndarray:
    """The same field with its largest-magnitude vectors (unstable estimates) set to zero."""
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    limit = np.percentile(magnitude, FLOW_TRIM_PERCENTILE)
    out = flow.copy()
    out[magnitude > limit] = 0.0
    return out


def flow_magnitude(flow: np.ndarray) -> float:
    """Mean magnitude after discarding the unstable top 2% of vectors."""
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1]).ravel()
    limit = np.percentile(magnitude, FLOW_TRIM_PERCENTILE)
    return float(magnitude[magnitude <= limit].mean())


def median_flow_magnitude(flow: np.ndarray) -> float:
    """Median magnitude: follows whole-frame motion and ignores a fast local subject."""
    return float(np.median(cv2.magnitude(flow[..., 0], flow[..., 1])))


def mean_flow(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Robust mean optical-flow magnitude in pixels between two grey frames."""
    return flow_magnitude(flow_field(prev_gray, gray))


def field_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two trimmed flow fields: near 1 when they move the same way."""
    a, b = trimmed_flow(a), trimmed_flow(b)
    num = float((a * b).sum())
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return num / den if den > 1e-6 else 0.0


def mean_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.absdiff(a, b).mean())
