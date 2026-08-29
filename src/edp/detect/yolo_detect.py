"""YOLO-based symbol localization — a drop-in replacement for
localize/proposals.py's skeleton-density candidate proposer on this
branch. See docs/02_model_selection_rationale.md for the rationale entry
and scripts/generate_synthetic_dataset.py / scripts/train_yolo.py for how
the detector is trained.

Deliberately still just a *localizer* here, not the final classifier:
YOLO's own class head is trained purely on synthetic composites (no real
schematic was in its training data), so its class confidence is a weaker
signal than the DINOv2+FAISS match against the same KiCad reference
library already used elsewhere in this pipeline — that comparison still
runs on every YOLO-proposed box via classify/match.py, unchanged. YOLO's
job is only "is there a symbol here at all," which is exactly the part
the density-based localizer struggled with (false positives on wire
bends, missed components — see the accuracy audit in docs/05).
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
    """Runs the trained detector and returns candidate bboxes.

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

    h, w = img_rgb.shape[:2]
    candidates: list[Candidate] = []
    for box in results[0].boxes:
        x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        bbox: BBox = (x0, y0, x1, y1)
        candidates.append(Candidate(bbox=bbox, kind="symbol"))

    # YOLO's own NMS (iou=cfg.yolo_iou_threshold above) doesn't catch every
    # near-duplicate: two boxes from different anchors/scales can each pass
    # NMS independently while still being 1-2px apart on the same physical
    # symbol. Observed concretely on D4: the same MOSFET, transistor, and
    # battery each boxed twice under different ids, which cascaded into
    # over-merged connectivity nets downstream (two near-identical boxes'
    # terminals both snapping to the same wire). Reuses the same merge the
    # density-based localizer needed for an analogous reason — see
    # localize/merge.py.
    return merge_overlapping(candidates, cfg.candidate_merge_overlap)
