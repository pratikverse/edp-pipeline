"""Merges heavily-overlapping candidate boxes into one.

Shared by both localizers (density-based in proposals.py, and the
experimental YOLO detector in detect/yolo_detect.py) since both can
independently produce two or three boxes for one physical symbol:
- density-based: density blobs for one symbol occasionally survive as
  separate, adjacent connected components (observed on D4's R4)
- YOLO: near-duplicate boxes from different anchors/scales that are close
  but not identical, so YOLO's own NMS doesn't collapse them (observed on
  D4's T4/IR540, boxed twice ~1-2px apart under different ids, which then
  cascaded into an over-merged connectivity net — see docs/02)

Merging keeps the higher-confidence duplicate's YOLO class/confidence
rather than discarding it: duplicates aren't always redundant noise, they
can be independent votes at different confidences. On D4, one duplicate
pair scored BJT_PNP 0.27 and BJT_NPN 0.53 for the same transistor — the
confident one was correct (see docs/08_improvement_plan.md section 2.2).
"""
from __future__ import annotations

from edp.types import BBox, Candidate


def merge_overlapping(candidates: list[Candidate], overlap_threshold: float) -> list[Candidate]:
    """Repeatedly unions any pair of candidates whose overlap relative to
    the smaller box exceeds the threshold, until no pair does."""
    boxes = list(candidates)
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap_ratio(boxes[i].bbox, boxes[j].bbox) >= overlap_threshold:
                    merged_bbox = _union_bbox(boxes[i].bbox, boxes[j].bbox)
                    kind = boxes[i].kind if boxes[i].kind == boxes[j].kind else "ambiguous"
                    winner = _higher_confidence(boxes[i], boxes[j])
                    boxes[i] = Candidate(
                        bbox=merged_bbox,
                        kind=kind,
                        yolo_class=winner.yolo_class,
                        yolo_confidence=winner.yolo_confidence,
                    )
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _higher_confidence(a: Candidate, b: Candidate) -> Candidate:
    conf_a = a.yolo_confidence if a.yolo_confidence is not None else -1.0
    conf_b = b.yolo_confidence if b.yolo_confidence is not None else -1.0
    return a if conf_a >= conf_b else b


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
