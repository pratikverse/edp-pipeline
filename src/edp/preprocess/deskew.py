"""Deskew via minAreaRect on foreground pixels. Small-angle correction only —
these are digitally generated drawings (docs/05), not scanned pages, so large
rotations are out of scope by design."""
from __future__ import annotations

import cv2
import numpy as np

from edp.config import PreprocessConfig


def estimate_skew_deg(binary: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # cv2.minAreaRect returns an angle in [-90, 0); normalise to a small
    # correction around 0 rather than snapping to the nearest 90 degrees.
    if angle < -45:
        angle = 90 + angle
    return angle


def deskew(binary: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, float]:
    angle = estimate_skew_deg(binary)
    if abs(angle) < 0.1 or abs(angle) > cfg.deskew_max_angle_deg:
        return binary, 0.0
    h, w = binary.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return rotated, angle
