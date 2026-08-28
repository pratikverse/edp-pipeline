"""Grayscale, adaptive binarize, denoise. Stage 1 of docs/01_architecture_overview.md."""
from __future__ import annotations

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
