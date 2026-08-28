"""Candidate -> Symbol: embed each candidate crop, nearest-neighbour match
against the reference library, threshold for "unknown". See docs/01 stage 3.
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
        entry, similarity = library.match(embeddings[i]) if len(embeddings) else (None, 0.0)

        if entry is None or similarity < cfg.unknown_similarity_threshold:
            symbol_type = UNKNOWN_TYPE
            rotation = 0
            confidence = similarity
        else:
            symbol_type = entry.class_name
            rotation = entry.rotation
            confidence = similarity

        symbol_id = f"SYM_{i:03d}"
        terminals = (
            _instantiate_terminals(symbol_id, entry, candidate.bbox)
            if entry is not None and symbol_type != UNKNOWN_TYPE
            else []
        )
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
