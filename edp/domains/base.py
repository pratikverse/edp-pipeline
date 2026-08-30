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
        )


def list_domains() -> list[str]:
    return sorted(
        p.parent.name
        for p in _DOMAINS_DIR.glob("*/pack.yaml")
        if p.parent.name != "__pycache__"
    )
