"""Candidate symbol proposal via skeleton branch/endpoint density.

Why not thick-vs-thin morphology (the original design): D4/D5 draw symbols
and wires with the *same* stroke width, so eroding to separate "thick" from
"thin" regions either erases everything (thin drawings like D5) or barely
separates anything (D4) — verified empirically, not assumed. What actually
distinguishes a symbol from a wire run in these drawings is *local
complexity*: a resistor zigzag, an IC box, a transistor's converging leads
all pack several skeleton corners/branches into a small area, while a wire
is mostly a long run of straight degree-2 skeleton pixels. See
docs/01_architecture_overview.md stage 2.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from edp.config import LocalizeConfig
from edp.types import BBox, Candidate, Point, TextToken

from .morphology import thin_stroke_mask

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def _strip_text(binary: np.ndarray, text_tokens: list[TextToken] | None, pad: int = 2) -> np.ndarray:
    """Text glyphs have just as many skeleton corners/endpoints as a real
    symbol, so without this, every label becomes a false candidate (see
    debug run during Day 1 tuning). OCR tokens are optional — if none are
    supplied, localization simply runs uncorrected for that noise source."""
    if not text_tokens:
        return binary
    stripped = binary.copy()
    h, w = binary.shape[:2]
    for token in text_tokens:
        x0, y0, x1, y1 = token.bbox
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        stripped[y0:y1, x0:x1] = 0
    return stripped


def _strip_dots(binary: np.ndarray, dots: list[Point], radius_pad: int = 3) -> np.ndarray:
    """A junction dot is a small solid disk at a branch point — locally
    just as "complex" as a real symbol, so without this it gets absorbed
    into whichever nearby symbol's density blob it's closest to, dragging
    that candidate's bbox out to include the dot and the wire stub
    leading to it (verified: this was why R1 in D5 cropped mostly wire and
    two junction dots, with only a sliver of the actual resistor).
    Junction dots are detected once, upstream, since the same detection
    is needed again in wires/junctions.py for connectivity."""
    if not dots:
        return binary
    stripped = binary.copy()
    for dx, dy in dots:
        cv2.circle(stripped, (dx, dy), radius_pad, 0, thickness=-1)
    return stripped


def _interest_points(binary: np.ndarray) -> np.ndarray:
    """Skeleton pixels that are a branch (degree>=3) or an endpoint
    (degree==1) — the raw "corner" signal everything else in this module
    is built from."""
    skeleton = skeletonize(binary > 0).astype(np.uint8)
    degree = convolve(skeleton, _NEIGHBOR_KERNEL, mode="constant", cval=0) * skeleton
    return ((degree >= 3) | (degree == 1)).astype(np.uint8)


def _density_map(interest_points: np.ndarray, window: int) -> np.ndarray:
    return cv2.boxFilter(interest_points.astype(np.float32), -1, (window, window), normalize=False)


def _corner_count(interest_points: np.ndarray, bbox: BBox) -> int:
    """Number of *distinct* corners within a bbox, not raw interest-pixel
    count: a single branch point can mark several adjacent skeleton
    pixels, which would otherwise inflate one corner into several. This
    is the direct fix for the dominant false-positive pattern — a plain
    wire bend or T-junction has exactly one corner and was being kept as
    a candidate by the windowed density check alone; a real symbol
    (zigzag, circle-with-leads, box) clusters several. Confirmed on D4:
    every rail-bend false positive audited had a single corner in its
    final bbox, while every real symbol had 3 or more.
    """
    x0, y0, x1, y1 = bbox
    region = interest_points[y0:y1, x0:x1]
    if region.size == 0:
        return 0
    num_labels, _ = cv2.connectedComponents(region, connectivity=8)
    return num_labels - 1  # exclude background label


def find_candidates(
    binary: np.ndarray,
    cfg: LocalizeConfig,
    text_tokens: list[TextToken] | None = None,
    dots: list[Point] | None = None,
) -> list[Candidate]:
    """Returns candidate bounding boxes, area-filtered.

    `kind` is a coarse hint (symbol/wire/ambiguous) based on area relative
    to the wire mask, used to prioritise stage-3 classification attempts —
    it is not a final decision, stage 3 owns that.

    Known limitation: dashed-outline symbols with sparse geometry (D5's
    SHIELD boundary) sit right at the density threshold and are unreliable.
    Tracked in docs/05.
    """
    clean = _strip_dots(_strip_text(binary, text_tokens), dots)
    density = _density_map(clean, window=cfg.density_window)

    mask = (density >= cfg.density_threshold).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((cfg.density_merge_kernel, cfg.density_merge_kernel), np.uint8))

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    wire_mask = thin_stroke_mask(binary, cfg)

    candidates: list[Candidate] = []
    pad = cfg.candidate_bbox_pad
    h, w = binary.shape[:2]
    for label in range(1, num_labels):
        x, y, cw, ch, _area = (int(v) for v in stats[label])
        loose_bbox: BBox = (max(0, x - pad), max(0, y - pad), min(w, x + cw + pad), min(h, y + ch + pad))
        bbox = _tighten_to_ink(clean, loose_bbox, pad=pad)
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if box_area < cfg.min_component_area or box_area > cfg.max_component_area:
            continue

        region_wire_frac = _wire_fraction(wire_mask, bbox)
        kind = "symbol" if region_wire_frac < 0.5 else "ambiguous"
        candidates.append(Candidate(bbox=bbox, kind=kind))

    return _merge_overlapping(candidates, cfg.candidate_merge_overlap)


def _merge_overlapping(candidates: list[Candidate], overlap_threshold: float) -> list[Candidate]:
    """Density blobs for one physical symbol occasionally survive as two
    or three separate, heavily-overlapping candidates rather than one
    (observed on D4's R4: three stacked boxes over a single resistor,
    each independently classified — wasted classification calls, and a
    JSON with the same physical symbol appearing three times under
    different ids). Repeatedly unions any pair whose overlap relative to
    the smaller box exceeds the threshold, until no pair does.
    """
    boxes = list(candidates)
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap_ratio(boxes[i].bbox, boxes[j].bbox) >= overlap_threshold:
                    merged_bbox = _union_bbox(boxes[i].bbox, boxes[j].bbox)
                    kind = boxes[i].kind if boxes[i].kind == boxes[j].kind else "ambiguous"
                    boxes[i] = Candidate(bbox=merged_bbox, kind=kind)
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _overlap_ratio(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    smaller_area = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _union_bbox(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _tighten_to_ink(binary: np.ndarray, loose_bbox: BBox, pad: int) -> BBox:
    """Shrinks a density-blob bbox down to the actual ink pixels it
    contains: the density window/merge-dilation inflate the box well
    beyond the true symbol extent. Bounded by construction — a stray wire
    running through the loose region can only pull the tightened box back
    out to `loose_bbox`, never further, unlike a full-image connectivity
    approach (see morphology.py's rejected thick/thin split)."""
    x0, y0, x1, y1 = loose_bbox
    h, w = binary.shape[:2]
    region = binary[y0:y1, x0:x1]
    ys, xs = np.where(region > 0)
    if len(xs) == 0:
        return loose_bbox
    tx0, tx1 = x0 + int(xs.min()), x0 + int(xs.max()) + 1
    ty0, ty1 = y0 + int(ys.min()), y0 + int(ys.max()) + 1
    return (
        max(0, tx0 - pad),
        max(0, ty0 - pad),
        min(w, tx1 + pad),
        min(h, ty1 + pad),
    )


def _wire_fraction(wire_mask: np.ndarray, bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    region = wire_mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region)) / region.size
