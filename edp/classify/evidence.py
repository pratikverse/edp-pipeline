"""Common representation for one classification signal, and confidence-
aware fusion across however many of them fired for a given candidate.

Why this exists as its own module rather than being folded straight into
match.py: docs/08_improvement_plan.md's classification architecture adds
independent evidence sources over time (visual, OCR text, and eventually
geometry specialists per section on confusion pairs) and none of them
should be allowed to just overwrite each other's answer — a loud, wrong
OCR read shouldn't clobber a confident visual match, and a confident
visual match shouldn't be immune to a genuinely strong OCR override (an
exact part-number hit). Keeping one small, source-agnostic type lets
match.py stay the orchestrator instead of every new source needing its
own bespoke merge logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassificationEvidence:
    """One source's opinion about a candidate's class.

    `class_scores` need not cover every class or sum to 1 — a source that
    has nothing to say returns an empty dict (see `NO_EVIDENCE`) rather
    than being forced to guess. `confidence` is the source's own estimate
    of how much to trust this particular reading (e.g. an exact
    part-number match is far more trustworthy than a bare one-letter
    designator, even though both are "text" evidence) — fusion weights by
    this, not just by a fixed per-source weight.
    """

    source: str  # "visual" | "text_prior" | "geometry:<specialist_name>" | ...
    class_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def has_evidence(self) -> bool:
        return bool(self.class_scores) and self.confidence > 0


def no_evidence(source: str, **metadata) -> ClassificationEvidence:
    """The explicit "this source abstained" value, so callers can tell
    "ambiguous OCR, ignore it" apart from "OCR confidently said nothing
    matches" — both currently look like an empty dict, but the metadata
    (e.g. `reason`) keeps the fusion trace explainable either way."""
    return ClassificationEvidence(source=source, class_scores={}, confidence=0.0, metadata=metadata)


@dataclass
class FusionResult:
    top_class: str
    top_score: float
    candidates: list[tuple[str, float]]  # sorted desc, top-3 kept
    margin: float  # top-1 score minus top-2 score (1.0 if only one candidate)
    trace: dict  # {source_name: {"class_scores": ..., "confidence": ..., "weight": ..., "metadata": ...}}


def fuse_evidence(
    evidences: list[ClassificationEvidence],
    weights: dict[str, float],
    default_weight: float = 1.0,
    unknown_type: str = "Unknown",
) -> FusionResult:
    """Weighted sum of each source's class_scores, scaled by that source's
    own confidence and a configured per-source base weight — so a source
    contributes `weight[source] * evidence.confidence * evidence.class_scores`
    rather than either the raw score alone (which would let a low-
    confidence reading vote as loudly as a certain one) or a fixed weight
    alone (which would let every reading from a source vote the same
    regardless of how sure it was this time).

    Sources with no evidence are skipped entirely rather than diluting the
    vote with zeros — see `ClassificationEvidence.has_evidence`.
    """
    totals: dict[str, float] = {}
    trace: dict = {}

    for ev in evidences:
        w = weights.get(ev.source, default_weight)
        trace[ev.source] = {
            "class_scores": dict(ev.class_scores),
            "confidence": ev.confidence,
            "weight": w,
            "metadata": ev.metadata,
        }
        if not ev.has_evidence:
            continue
        for class_name, score in ev.class_scores.items():
            totals[class_name] = totals.get(class_name, 0.0) + w * ev.confidence * score

    if not totals:
        return FusionResult(top_class=unknown_type, top_score=0.0, candidates=[], margin=1.0, trace=trace)

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # Margin is reported on the *normalised* scale (fraction of the winning
    # score) so it means the same thing regardless of how many sources fired
    # or how large their raw weighted totals happen to be.
    margin = 1.0 if top_score == 0 else (top_score - second_score) / top_score

    return FusionResult(
        top_class=top_class,
        top_score=top_score,
        candidates=ranked[:3],
        margin=margin,
        trace=trace,
    )
