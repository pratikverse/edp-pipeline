"""Stage 1 - preprocessing: grayscale, adaptive binarization, deskew,
speckle denoise, optional blue/black colour-layer separation."""
from __future__ import annotations


# ===========================================================================
# binarize.py
# ===========================================================================

import cv2
import numpy as np

from edp.config import PreprocessConfig


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Returns a binary image: 255 = ink (foreground), 0 = background.

    Adaptive threshold rather than a single global threshold — line-art
    drawings can have uneven scan/render background, and adaptive
    thresholding is robust to that without needing per-drawing tuning.
    """
    block = cfg.binarize_block_size | 1  # must be odd
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        cfg.binarize_c,
    )
    if cfg.denoise:
        binary = _remove_speckles(binary, min_area=cfg.denoise_min_speckle_area)
    return binary


def _remove_speckles(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Drops connected components smaller than min_area (isolated
    thresholding noise), leaving continuous strokes untouched regardless
    of width.

    A blanket morphological opening was tried first and rejected: on
    these drawings even a 2x2 open kernel erases every 1px-wide wire and
    symbol stroke outright (a 1px line cannot contain a 2x2 structuring
    element anywhere along its length), while leaving junction dots
    intact — verified by inspecting the binarized D5 output, where nearly
    all line art vanished and only dots/text fragments survived.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned

# ===========================================================================
# deskew.py
# ===========================================================================

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

# ===========================================================================
# layers.py
# ===========================================================================

import cv2
import numpy as np

# Loose blue range covering typical schematic "blue wire" rendering.
_BLUE_LOWER = np.array([90, 40, 40])
_BLUE_UPPER = np.array([140, 255, 255])


def blue_layer_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Returns a binary mask (255 = blue ink) same size as input.

    All-zero on monochrome drawings (e.g. D4) — callers must treat this as
    "no colour signal available", not as an error.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] < 3:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)
