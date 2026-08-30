"""Domain packs — the seam between the domain-agnostic pipeline machinery
and the per-drawing-type knowledge it consults.

The pipeline stages (preprocess, localize, classify, ocr, wires, emit) are
identical for an electronic schematic and a P&ID. What differs is *what
they know*: which symbol library to match against, which trained detector
to run, which OCR text patterns mean which component, which geometry
specialists resolve which confusion pairs, and which connectivity
conventions apply. A `DomainPack` bundles exactly that, so adding a new
drawing type is a new `edp/domains/<name>/` folder, not a change to any
stage.

Each pack is `edp/domains/<name>/pack.yaml` plus, alongside it, whatever
small knowledge files that pack references (a `specialists.py`, a
`designators.yaml`). Large regeneratable assets (reference-image
libraries, trained weights) stay under `data/` and are pointed at by
path from `pack.yaml`.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

from edp.realdetect import RealDetectorSpec

_DOMAINS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DOMAINS_DIR.parents[1]


@dataclass(frozen=True)
class DomainPack:
    """One drawing type's knowledge. Paths are resolved to absolute at load."""

    name: str
    detector_weights: Path          # trained YOLO for this domain
    reference_dir: Path             # symbol library: <class>/*.png + *.terminals.json
    designators_path: Path          # OCR designator / part-number table (YAML)
    probe_model_path: Path | None   # linear-probe artifact; None -> source abstains
    evidence_weights: dict[str, float] = field(default_factory=dict)
    specialists_module: str | None = None  # dotted path exposing CONFUSION_GROUPS + select_specialist
    real_detector: "RealDetectorSpec | None" = None  # off-the-shelf recall-booster detector
    router_keywords: list[str] = field(default_factory=list)  # page-router hints (see route())

    @cached_property
    def _specialists(self):
        if not self.specialists_module:
            return None
        return importlib.import_module(self.specialists_module)

    @property
    def confusion_groups(self) -> list[frozenset[str]]:
        mod = self._specialists
        return list(getattr(mod, "CONFUSION_GROUPS", [])) if mod else []

    def select_specialist(self, candidate_classes: set[str]):
        mod = self._specialists
        if mod is None:
            return None
        return mod.select_specialist(candidate_classes)

    @classmethod
    def load(cls, name: str, repo_root: Path | None = None) -> "DomainPack":
        root = repo_root or _REPO_ROOT
        pack_yaml = _DOMAINS_DIR / name / "pack.yaml"
        if not pack_yaml.exists():
            raise FileNotFoundError(
                f"unknown domain '{name}': no {pack_yaml.relative_to(_REPO_ROOT)} "
                f"(available: {', '.join(list_domains()) or 'none'})"
            )
        d = yaml.safe_load(pack_yaml.read_text(encoding="utf-8")) or {}

        def _p(key: str) -> Path:
            raw = Path(d[key])
            return raw if raw.is_absolute() else (root / raw)

        probe = d.get("probe_model_path")
        return cls(
            name=name,
            detector_weights=_p("detector_weights"),
            reference_dir=_p("reference_dir"),
            designators_path=_p("designators_path"),
            probe_model_path=(root / probe) if probe else None,
            evidence_weights=dict(d.get("evidence_weights", {})),
            specialists_module=d.get("specialists_module"),
            real_detector=RealDetectorSpec.from_dict(d.get("real_detector")),
            router_keywords=[k.upper() for k in d.get("router_keywords", [])],
        )

    def _designator_prefixes(self) -> set[str]:
        try:
            raw = yaml.safe_load(self.designators_path.read_text(encoding="utf-8")) or {}
            return {k.upper() for k in (raw.get("designators") or {})}
        except Exception:
            return set()


def list_domains() -> list[str]:
    return sorted(
        p.parent.name
        for p in _DOMAINS_DIR.glob("*/pack.yaml")
        if p.parent.name != "__pycache__"
    )


def route(gray, text_cfg, default: str = "electronic") -> tuple[str, dict]:
    """Pick a domain pack for a drawing from its text alone.

    A P&ID and an electronic schematic are told apart cheaply and
    reliably by what's written on them: process words and ISA instrument
    tags (GPM, OUTLET, PC, LT, FCV) vs. component values, units and part
    numbers (10K, uF, BC547, R6). Each pack contributes its designator
    prefixes plus a `router_keywords` list; the drawing's OCR tokens are
    scored against both and the higher score wins. Falls back to
    `default` when the signal is thin or tied, so a text-sparse drawing
    still runs rather than erroring.

    Returns (domain_name, scores, tokens) — the OCR tokens are handed back
    so the caller doesn't run OCR twice.
    """
    import re as _re

    from edp.ocr import run_ocr

    tokens = run_ocr(gray, text_cfg)
    norm = [_re.sub(r"[^A-Z0-9Ω°ΜµμΩ]", "", t.text.upper()) for t in tokens]
    norm = [t for t in norm if t]

    scores: dict[str, int] = {}
    for name in list_domains():
        pack = DomainPack.load(name)
        prefixes = pack._designator_prefixes()
        kws = pack.router_keywords
        s = 0
        for tok in norm:
            if any(tok.startswith(p) and tok[len(p):][:1].isdigit() for p in prefixes if p):
                s += 1
            if any(kw in tok for kw in kws):
                s += 1
        scores[name] = s

    best = max(scores, key=scores.get) if scores else default
    if not scores or scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        best = default
    return best, scores, tokens
