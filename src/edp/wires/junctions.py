"""Junction-dot detection: filled circle at an intersection = connected,
bare crossing = not connected. See docs/01 stage 6 and docs/06_data_model.md
("crossing decomposition" — the core connectivity-correctness rule)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from edp.config import WiresConfig
from edp.types import Point

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def detect_junction_dots(binary: np.ndarray, cfg: WiresConfig) -> list[Point]:
    """Finds filled dots by local fill-ratio at skeleton branch points,
    not by contour shape.

    A `cv2.findContours` approach was tried first and rejected: a dot
    sitting directly on a wire (the normal case — a dot only exists where
    wires meet) is 8-connected to that wire, so the whole dot+wire run
    becomes one contour, and `minEnclosingCircle` of that shape is nothing
    like a small circle. Verified this was silently dropping most real
    dots (only ones forming their own isolated contour survived — 30 out
    of many more actually present in D5). Checking local ink density in a
    small disk around every skeleton branch point instead is invariant to
    what else that point happens to be connected to: a plain crossing
    fills only the width of the two crossing strokes, a drawn dot fills
    most of the disk.
    """
    skeleton = skeletonize(binary > 0).astype(np.uint8)
    degree = convolve(skeleton, _NEIGHBOR_KERNEL, mode="constant", cval=0) * skeleton
    branch_ys, branch_xs = np.where(degree >= 3)

    # radius=max_radius diluted fill ratio far below threshold (a ~2px dot
    # fills only ~16% of a radius-5 disk); radius=min_radius overfired on
    # ordinary zigzag/coil corners, which are locally dense enough to pass
    # a tiny disk's fill check too. max_radius-1 was the empirical sweet
    # spot verified by eye against D5: catches dots at real junctions
    # without lighting up every symbol corner.
    radius = cfg.junction_dot_max_radius - 1
    disk_area = np.pi * radius * radius
    h, w = binary.shape[:2]

    dots: list[Point] = []
    for x, y in zip(branch_xs.tolist(), branch_ys.tolist()):
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(w, x + radius + 1), min(h, y + radius + 1)
        disk_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(disk_mask, (x - x0, y - y0), radius, 1, thickness=-1)
        region = binary[y0:y1, x0:x1] > 0
        fill_ratio = float(np.count_nonzero(region & (disk_mask > 0))) / disk_area
        if fill_ratio > cfg.junction_dot_fill_ratio:
            dots.append((x, y))
    return dots


def is_dotted_crossing(point: Point, dots: list[Point], tolerance: int = 4) -> bool:
    px, py = point
    return any(abs(px - dx) <= tolerance and abs(py - dy) <= tolerance for dx, dy in dots)
