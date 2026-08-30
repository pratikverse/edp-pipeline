"""YOLO-based symbol localization — a drop-in replacement for
localize/proposals.py's skeleton-density candidate proposer on this
branch. See docs/02_model_selection_rationale.md for the rationale entry
and scripts/generate_synthetic_dataset.py / scripts/train_yolo.py for how
the detector is trained.

Also carries YOLO's own class prediction (`box.cls`/`box.conf`) through
on each Candidate for classify/match.py to fuse with the DINOv2+FAISS
match, rather than discarding it. An earlier version of this module
treated YOLO as localization-only, reasoning its class head was trained
purely on synthetic composites and so was a weaker signal than DINOv2.
That reasoning didn't hold up: YOLO is *supervised* on our exact class
set and has seen schematic-style line art in training, where DINOv2 is
self-supervised on natural photographs and has never seen a schematic at
all. Measured on D4: YOLO's top-1 class was correct on ~10/16 checkable
symbols vs. DINOv2's ~7/16, and the two disagree on different symbols
(YOLO gets the transistor polarities and R3/R6/C1 right where DINOv2
doesn't; DINOv2 gets S1/T4 right where YOLO doesn't) — see
docs/08_improvement_plan.md section 2.1 for the full comparison.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from edp.config import LocalizeConfig
from edp.localize.merge import merge_overlapping
from edp.types import BBox, Candidate


@lru_cache(maxsize=1)
def _load_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def detect_candidates(img_rgb: np.ndarray, cfg: LocalizeConfig) -> list[Candidate]:
    """Runs the trained detector and returns candidate bboxes, each
    carrying YOLO's own top-1 class/confidence for the fusion policy in
    classify/match.py.

    Falls back to an empty candidate list (not an exception) if the
    weights file doesn't exist yet — keeps `edp run` usable while a
    training run is still in progress, matching the same "stay runnable
    with an incomplete model" principle as the empty-reference-library
    case in classify/library.py.
    """
    weights_path = Path(cfg.yolo_weights)
    if not weights_path.exists():
        return []

    model = _load_model(str(weights_path))
    results = model.predict(img_rgb, conf=cfg.yolo_conf_threshold, iou=cfg.yolo_iou_threshold, verbose=False)
    class_names = results[0].names

    h, w = img_rgb.shape[:2]
    candidates: list[Candidate] = []
    for box in results[0].boxes:
        x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        bbox: BBox = (x0, y0, x1, y1)
        yolo_class = class_names[int(box.cls[0])]
        yolo_confidence = float(box.conf[0])
        candidates.append(
            Candidate(bbox=bbox, kind="symbol", yolo_class=yolo_class, yolo_confidence=yolo_confidence)
        )

    # YOLO's own NMS (iou=cfg.yolo_iou_threshold above) doesn't catch every
    # near-duplicate: two boxes from different anchors/scales can each pass
    # NMS independently while still being 1-2px apart on the same physical
    # symbol. Observed concretely on D4: the same MOSFET, transistor, and
    # battery each boxed twice under different ids, which cascaded into
    # over-merged connectivity nets downstream (two near-identical boxes'
    # terminals both snapping to the same wire). Reuses the same merge the
    # density-based localizer needed for an analogous reason — see
    # localize/merge.py. The merge keeps the higher-confidence duplicate's
    # class rather than dropping class info: on D4, one duplicate pair
    # scored BJT_PNP 0.27 and BJT_NPN 0.53 for the same transistor, and the
    # confident one was correct.
    return merge_overlapping(candidates, cfg.candidate_merge_overlap)
