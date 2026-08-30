"""Symbol subtraction, skeletonization, skeleton-graph construction.
See docs/01 stage 5."""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

from edp.types import Symbol


def subtract_symbols(binary: np.ndarray, symbols: list[Symbol], pad: int = 2) -> np.ndarray:
    """Zeroes out symbol bboxes so only wire pixels remain."""
    wire_only = binary.copy()
    h, w = binary.shape[:2]
    for s in symbols:
        x0, y0, x1, y1 = s.bbox
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        wire_only[y0:y1, x0:x1] = 0
    return wire_only


def skeletonize_wires(wire_mask: np.ndarray) -> np.ndarray:
    """Morphological thinning to 1px-wide skeleton. Returns uint8 0/255."""
    bool_mask = wire_mask > 0
    thin = skeletonize(bool_mask)
    return (thin.astype(np.uint8)) * 255
