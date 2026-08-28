"""Colour-layer separation. D5 draws some conductors in blue; D4 is
monochrome. This is an opportunistic signal, never a hard dependency
(docs/02_model_selection_rationale.md) — every caller must handle
color_mask being all-zero.
"""
from __future__ import annotations

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
