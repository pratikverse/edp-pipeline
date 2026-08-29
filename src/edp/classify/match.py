"""Candidate -> Symbol: fuses independent classification evidence sources
(YOLO's own class prediction, DINOv2+FAISS top-k reference match, and an
OCR reference-designator/part-number prior) via classify/evidence.py's
weighted fusion, instead of the earlier ad hoc rule-based YOLO/DINOv2
policy. See docs/08_improvement_plan.md for the evidence architecture and
why: a fixed priority order ("OCR > DINO > YOLO") lets one loud, wrong
source clobber two right ones, where a confidence-weighted vote lets each
source's *own* certainty decide how much it counts this time.

Each source is independent and may abstain (`ClassificationEvidence` with
no scores) rather than being forced to guess:

  - "yolo": the localizer's own class + confidence, when the YOLO
    localizer produced the candidate (see detect/yolo_detect.py). None
    for the density-based localizer.
  - "dinov2": top-k *distinct classes* from the reference-embedding
    match (classify/library.py's match_topk), each entry's cosine
    similarity as its score, entries below unknown_similarity_threshold
    dropped so a bad match can't force a wrong answer through.
  - "text_prior": the OCR reference-designator/part-number prior
    (classify/text_prior.py), gated by how specific the OCR match was —
    an exact part number is near-authoritative, a bare designator is a
    moderate vote, unparseable OCR contributes nothing at all.

The historical D4 numbers motivating this (docs/08 section 2.1): YOLO
~10/16 correct, DINOv2 ~7/16, on largely non-overlapping error sets — so
combining beats trusting either alone, and OCR closes several of the
errors *both* miss (a part number like "BC547" is unambiguous evidence
neither a supervised detector trained on symbol shape nor an unsupervised
embedding match can see).
"""
from __future__ import annotations

import math

import numpy as np

from edp.config import ClassifyConfig
from edp.types import Candidate, Symbol, Terminal

from .embedder import Embedder
from .evidence import ClassificationEvidence, fuse_evidence, no_evidence
from .library import LibraryEntry, ReferenceLibrary
from .text_prior import evidence_from_text

UNKNOWN_TYPE = "Unknown"

DEFAULT_EVIDENCE_WEIGHTS = {
    # YOLO gets a modest edge over DINOv2 at equal confidence -- it's the
    # supervised, in-domain signal (trained on our own schematic symbols),
    # where DINOv2 is self-supervised on natural photographs and has never
    # seen a schematic (docs/08 section 2.1). Both are still genuine votes,
    # not a hard override -- a very confident DINOv2 match can still win
    # against a weak YOLO one.
    "yolo": 1.2,
    "dinov2": 1.0,
    "text_prior": 1.0,
}


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
    """Canonical (rotation=0, unmirrored) reference entry for `class_name`,
    used only when the winning class didn't come from the DINOv2 top-k at
    all (e.g. text_prior or YOLO alone picked it) — so there's no
    already-matched, rotation-aware entry to reuse for the terminal
    template. Falls back to any entry of that class if no canonical one
    exists."""
    fallback = None
    for entry in library.entries:
        if entry.class_name != class_name:
            continue
        if entry.rotation == 0 and not entry.mirrored:
            return entry
        fallback = fallback or entry
    return fallback


def _yolo_evidence(candidate: Candidate) -> ClassificationEvidence:
    if candidate.yolo_class is None:
        return no_evidence("yolo", reason="density_localizer_no_yolo_class")
    return ClassificationEvidence(
        source="yolo",
        class_scores={candidate.yolo_class: 1.0},
        confidence=candidate.yolo_confidence or 0.0,
        metadata={"yolo_class": candidate.yolo_class, "yolo_confidence": candidate.yolo_confidence},
    )


def _dinov2_evidence(
    embedding: np.ndarray, library: ReferenceLibrary, cfg: ClassifyConfig
) -> tuple[ClassificationEvidence, list[tuple[str, float, LibraryEntry]]]:
    topk = library.match_topk(embedding, k=3) if len(library) else []
    above_threshold = [(cn, score, entry) for cn, score, entry in topk if score >= cfg.unknown_similarity_threshold]
    if not above_threshold:
        return no_evidence("dinov2", reason="no_match_above_threshold", raw_topk=[(cn, round(s, 3)) for cn, s, _ in topk]), topk
    class_scores = {cn: score for cn, score, _ in above_threshold}
    ev = ClassificationEvidence(
        source="dinov2",
        class_scores=class_scores,
        confidence=1.0,  # each score is already a meaningful cosine similarity, not a raw vote to discount further
        metadata={"topk": [(cn, round(s, 3)) for cn, s, _ in topk]},
    )
    return ev, topk


def classify_candidates(
    img_rgb: np.ndarray,
    candidates: list[Candidate],
    library: ReferenceLibrary,
    cfg: ClassifyConfig,
    embedder: Embedder | None = None,
    ocr_hints: list[str] | None = None,
) -> list[Symbol]:
    """`ocr_hints[i]`, when provided, is the OCR text already gathered near
    candidates[i].bbox *before* classification (pipeline.py computes this
    via text.associate.nearby_token_text, since classification runs ahead
    of the final text-association stage — see pipeline.py's comment on
    stage order). Passing None (or a list of empty strings) degrades
    cleanly to text_prior abstaining on every candidate."""
    embedder = embedder or Embedder(cfg.model)

    crops = [_crop(img_rgb, c.bbox) for c in candidates]
    embeddings = embedder.embed(crops) if crops else np.zeros((0, cfg.embedding_dim))
    hints = ocr_hints or [None] * len(candidates)

    weights = {**DEFAULT_EVIDENCE_WEIGHTS, **(cfg.evidence_weights or {})}

    symbols: list[Symbol] = []
    for i, candidate in enumerate(candidates):
        dinov2_ev, dinov2_topk = _dinov2_evidence(embeddings[i], library, cfg) if len(embeddings) else (
            no_evidence("dinov2", reason="empty_library"),
            [],
        )
        yolo_ev = _yolo_evidence(candidate)
        text_ev = evidence_from_text(hints[i], cfg.reference_designators_path)

        fusion = fuse_evidence([yolo_ev, dinov2_ev, text_ev], weights, unknown_type=UNKNOWN_TYPE)
        symbol_type = fusion.top_class
        confidence = fusion.top_score

        # Terminal template: prefer the actual DINOv2 top-k entry for the
        # winning class (rotation/mirror-aware, since match_topk returns
        # whichever augmented variant scored highest) over a blind
        # canonical lookup — see _find_entry_for_class's docstring.
        entry = next((e for cn, _s, e in dinov2_topk if cn == symbol_type), None)
        if entry is None and symbol_type != UNKNOWN_TYPE:
            entry = _find_entry_for_class(library, symbol_type)

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
                top_k=fusion.candidates,
                margin=fusion.margin,
                evidence_trace=fusion.trace,
            )
        )
    return symbols


def _crop(img_rgb: np.ndarray, bbox) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return img_rgb[y0:y1, x0:x1]
