"""OCR text -> classification evidence (docs/08_improvement_plan.md,
classification evidence architecture). Turns the OCR text already
associated near a candidate into a `ClassificationEvidence`, deterministic
and config-driven — see config/reference_designators.yaml for the actual
designator/part-number tables; nothing here is hardcoded.

Deliberately does not duplicate OCR itself or symbol/text association —
this module only consumes text that text/ocr.py and text/associate.py
already produced (see pipeline.py for where the pre-classification text
hint comes from) and turns it into a scored prior.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .evidence import ClassificationEvidence, no_evidence

_ID_STRIP_RE = re.compile(r"[\s\-_.]+")
_DESIGNATOR_RE = re.compile(r"^([A-Z]+)(\d+)")

# Safe, unambiguous OCR character substitutions only -- a glyph that Tesseract
# is well known to confuse with a letter that would otherwise block the
# designator regex outright, never a substitution that could turn one valid
# reading into a different valid one (see normalize_ocr_text's docstring).
# Observed concretely on D4: "S1" (the switch designator) OCR'd as "$1",
# which fails _DESIGNATOR_RE (leading char isn't A-Z) and silently drops a
# case the table would otherwise resolve correctly.
_SAFE_SUBSTITUTIONS = {
    "$": "S",  # dollar sign has no legitimate meaning in a component label
}


class EvidenceLevel:
    """How specific the OCR match was, per docs/08's confidence-gating
    requirement — never conflate a bare designator with an exact part
    number, even though both come from the same source."""

    EXACT_PART_NUMBER = "exact_part_number"  # e.g. "BC548" -> confidence 0.95
    DESIGNATOR = "designator"  # e.g. "R6", "ZD1" -> confidence 0.55
    AMBIGUOUS = "ambiguous"  # unparseable / no match -> no evidence at all


@dataclass
class DesignatorTable:
    designators: dict[str, dict[str, float]]
    part_number_patterns: list[tuple[re.Pattern, str]]


@lru_cache(maxsize=1)
def _load_table(path: str) -> DesignatorTable:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    designators = {k.upper(): v for k, v in (raw.get("designators") or {}).items()}
    patterns = [
        (re.compile(entry["pattern"], re.IGNORECASE), entry["class"])
        for entry in raw.get("part_number_families", [])
    ]
    return DesignatorTable(designators=designators, part_number_patterns=patterns)


def normalize_ocr_text(raw: str) -> str:
    """Upper-cases and strips whitespace/punctuation noise a Tesseract read
    commonly introduces around a label ("R 6" -> "R6", "ZD-1" -> "ZD1"),
    plus a small, explicit table of unambiguous character substitutions
    (_SAFE_SUBSTITUTIONS) — deliberately not general character-level OCR
    error correction, which would risk turning a low-confidence read into
    a confident-looking wrong one. Every entry in that table has no other
    legitimate reading, which is what makes it "safe" rather than a guess."""
    text = _ID_STRIP_RE.sub("", raw.upper())
    for bad, good in _SAFE_SUBSTITUTIONS.items():
        text = text.replace(bad, good)
    return text


def evidence_from_text(raw_text: str | None, cfg_path: str = "config/reference_designators.yaml") -> ClassificationEvidence:
    """Returns text-derived classification evidence for one candidate's
    nearby OCR text (already gathered by the caller — see
    text/associate.py's shared nearest-token helper).

    Checks part-number families before designator prefixes: a part number
    is far more specific evidence than a one-letter prefix, so it wins
    when both are present in the same token (e.g. "T2BC547" matches
    designator "T" -- deliberately absent from the table, see the yaml
    comment -- and part family "BC547" -> BJT_NPN).

    `raw_text` is frequently several distinct OCR readings joined with
    spaces (text/associate.py's nearby_token_text gathers several nearby
    tokens, and since text/ocr.py's run_ocr runs multiple Tesseract
    configs per orientation without deduplicating -- see its docstring --
    some of those readings are noise fragments from a different config
    pass reading the same physical label badly). Part-number matching
    searches the whole normalised blob (a substring search doesn't care
    about leading junk), but designator matching checks each
    whitespace-separated token independently rather than the joined blob:
    an early version normalised-then-matched the *whole* joined string,
    which anchors at its start (`_DESIGNATOR_RE.match`) -- one junk
    fragment before the real reading ("= ae R6 R6 R6 R6" -> "=AER6...")
    was enough to silently defeat a correct, unambiguous "R6" sitting
    right next to it. Checking token-by-token is immune to that."""
    if not raw_text or not raw_text.strip():
        return no_evidence("text_prior", reason="no_ocr_text")

    table = _load_table(cfg_path)
    normalized = normalize_ocr_text(raw_text)

    for pattern, class_name in table.part_number_patterns:
        if pattern.search(normalized):
            return ClassificationEvidence(
                source="text_prior",
                class_scores={class_name: 1.0},
                confidence=0.95,
                metadata={"level": EvidenceLevel.EXACT_PART_NUMBER, "raw_text": raw_text, "normalized": normalized},
            )

    for raw_token in raw_text.split():
        token_normalized = normalize_ocr_text(raw_token)
        match = _DESIGNATOR_RE.match(token_normalized)
        if not match:
            continue
        prefix = match.group(1)
        # Longest matching prefix wins ("BATT" over "B", "XTAL" over "X")
        # so a longer, more specific designator isn't shadowed by a short
        # one that happens to also be a valid prefix.
        candidates = [p for p in table.designators if prefix.startswith(p)]
        if not candidates:
            # A designator-shaped token whose prefix isn't in the table at
            # all (e.g. "U1", "TP2") — genuinely nothing to say about
            # *this* token; keep checking the rest rather than giving up.
            continue
        best = max(candidates, key=len)
        return ClassificationEvidence(
            source="text_prior",
            class_scores=dict(table.designators[best]),
            confidence=0.55,
            metadata={
                "level": EvidenceLevel.DESIGNATOR,
                "raw_text": raw_text,
                "matched_token": raw_token,
                "designator": best,
            },
        )

    # No part-number hit and no token matched a recognised designator --
    # garbled OCR, a value string ("10K", "220R"), or an id we don't model
    # ("TP2"). Per docs/08's gating rule, this is ignored rather than
    # treated as weak evidence: never let unparseable OCR vote at all.
    return no_evidence("text_prior", reason=EvidenceLevel.AMBIGUOUS, raw_text=raw_text, normalized=normalized)
