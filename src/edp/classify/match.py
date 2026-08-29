"""Candidate -> Symbol: fuses YOLO's own class prediction (when present)
with the DINOv2+FAISS reference-library match, rather than trusting
either alone. See docs/08_improvement_plan.md section 2.1 for why: YOLO
is supervised on our exact class set and has seen schematic-style line
art in training; DINOv2 is self-supervised on natural photographs and
has never seen a schematic. Measured on D4 they get ~10/16 and ~7/16
symbols right respectively, on non-overlapping error sets — so fusing
beats trusting either one exclusively.

Fusion policy, in order:
1. Both agree -> that class, confidence = max of the two
2. Disagree, YOLO's confidence clears yolo_fusion_confidence_threshold
   -> YOLO's class (it's the in-domain, supervised signal)
3. Disagree, YOLO weak -> DINOv2's class, if it clears its own threshold
4. Neither clears its threshold -> Unknown

The density-based localizer (candidate.yolo_class is None) skips
straight to the DINOv2-only path — this is the pre-fusion behaviour,
unchanged for that localizer.
"""
from __future__ import annotations

import cv2
import numpy as np

from edp.config import ClassifyConfig
from edp.types import Candidate, Symbol, Terminal

from .embedder import Embedder
from .library import LibraryEntry, ReferenceLibrary

UNKNOWN_TYPE = "Unknown"


def _instantiate_terminals(symbol_id: str, entry: LibraryEntry, bbox) -> list[Terminal]:
    """Scales the matched reference entry's normalised terminal template
    onto the candidate's actual bbox — this is what turns a KiCad pin
    template into a real, image-space connection point (docs/06).

    Direction is recovered from tip-minus-body in the *scaled* space: since
    almost every KiCad pin is axis-aligned (0/90/180/270), independent x/y
    scaling from a non-square bbox doesn't distort the angle for those —
    only genuinely diagonal pins would skew slightly, none are in the
    current curated library.
    """
    import math

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    terminals = []
    for i, (name, (tfx, tfy), (bfx, bfy)) in enumerate(entry.terminals):
        tip = (x0 + round(tfx * w), y0 + round(tfy * h))
        body = (x0 + round(bfx * w), y0 + round(bfy * h))
        dx, dy = tip[0] - body[0], tip[1] - body[1]
        direction = math.degrees(math.atan2(dy, dx)) if (dx, dy) != (0, 0) else None
        terminals.append(
            Terminal(
                symbol_id=symbol_id,
                index=i,
                point=tip,
                name=name or None,
                source="library",
                direction_deg=direction,
            )
        )
    return terminals


def _find_entry_for_class(library: ReferenceLibrary, class_name: str) -> LibraryEntry | None:
    """Finds a rotation=0, non-mirrored reference entry for `class_name`,
    for terminal-template purposes when YOLO's chosen class differs from
    whatever the DINOv2 nearest-neighbour actually retrieved — the
    retrieved entry's terminals belong to *its* class, not YOLO's, so
    they can't be reused directly. Falls back to any entry of that class
    if no canonical (rotation=0, unmirrored) one exists."""
    fallback = None
    for entry in library.entries:
        if entry.class_name != class_name:
            continue
        if entry.rotation == 0 and not entry.mirrored:
            return entry
        fallback = fallback or entry
    return fallback


def _fuse(
    dinov2_type: str | None,
    dinov2_confidence: float,
    yolo_type: str | None,
    yolo_confidence: float | None,
    yolo_fusion_threshold: float,
) -> tuple[str, float, str]:
    """Returns (type, confidence, source) where source is one of
    "agree" | "yolo" | "dinov2" | "none", used only for picking which
    library entry backs the terminal template."""
    if dinov2_type is not None and yolo_type is not None and dinov2_type == yolo_type:
        return dinov2_type, max(dinov2_confidence, yolo_confidence or 0.0), "agree"

    if yolo_type is not None and (yolo_confidence or 0.0) >= yolo_fusion_threshold:
        return yolo_type, yolo_confidence, "yolo"

    if dinov2_type is not None:
        return dinov2_type, dinov2_confidence, "dinov2"

    if yolo_type is not None:
        # YOLO had *something* even though it didn't clear the fusion bar,
        # and DINOv2 had nothing usable at all — a weak signal beats none.
        return yolo_type, yolo_confidence or 0.0, "yolo"

    return UNKNOWN_TYPE, max(dinov2_confidence, yolo_confidence or 0.0), "none"


def classify_candidates(
    img_rgb: np.ndarray,
    candidates: list[Candidate],
    library: ReferenceLibrary,
    cfg: ClassifyConfig,
    embedder: Embedder | None = None,
) -> list[Symbol]:
    embedder = embedder or Embedder(cfg.model)

    crops = [_crop(img_rgb, c.bbox) for c in candidates]
    embeddings = embedder.embed(crops) if crops else np.zeros((0, cfg.embedding_dim))

    symbols: list[Symbol] = []
    for i, candidate in enumerate(candidates):
        dinov2_entry, dinov2_similarity = library.match(embeddings[i]) if len(embeddings) else (None, 0.0)
        dinov2_type = (
            dinov2_entry.class_name
            if dinov2_entry is not None and dinov2_similarity >= cfg.unknown_similarity_threshold
            else None
        )

        symbol_type, confidence, source = _fuse(
            dinov2_type,
            dinov2_similarity,
            candidate.yolo_class,
            candidate.yolo_confidence,
            cfg.yolo_fusion_confidence_threshold,
        )

        # Terminal template must come from a library entry of the *winning*
        # class — the DINOv2-retrieved entry only if that's the class that
        # actually won; otherwise look up any entry of the winning class.
        if source in ("agree", "dinov2") and dinov2_entry is not None:
            entry = dinov2_entry
        elif symbol_type != UNKNOWN_TYPE:
            entry = _find_entry_for_class(library, symbol_type)
        else:
            entry = None

        rotation = entry.rotation if entry is not None else 0
        symbol_id = f"SYM_{i:03d}"
        terminals = _instantiate_terminals(symbol_id, entry, candidate.bbox) if entry is not None else []

        symbols.append(
            Symbol(
                id=symbol_id,
                type=symbol_type,
                bbox=candidate.bbox,
                confidence=confidence,
                rotation=rotation,
                terminals=terminals,
                label_source="classification",
            )
        )
    return symbols


def _crop(img_rgb: np.ndarray, bbox) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return img_rgb[y0:y1, x0:x1]
