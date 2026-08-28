"""Reference embedding library: build from data/reference/<class>/*.png,
rotation/mirror-augmented, embedded once and cached to disk. See
docs/02_model_selection_rationale.md (rotation handling, style variants)
and docs/07_project_layout.md (`edp build-library`).

Terminal templates (data/reference/<class>/<name>.terminals.json, produced
by scripts/build_reference_from_kicad.py) are carried through the same
rotation/mirror augmentation as the image, in normalised [0,1] crop
fractions — see docs/06_data_model.md for why terminal-level connectivity
matters and classify/match.py for where they get scaled onto a candidate.
Each terminal keeps both its tip (wire contact point) and body (pin's
inner anchor) so the outward pin direction can be reconstructed after
scaling, for directional snapping in wires/nets.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from edp.config import ClassifyConfig

from .embedder import Embedder

TerminalTemplate = tuple[str, tuple[float, float], tuple[float, float]]  # name, tip_frac, body_frac


@dataclass
class LibraryEntry:
    class_name: str
    rotation: int
    mirrored: bool
    source_path: str
    embedding: np.ndarray
    terminals: list[TerminalTemplate] = field(default_factory=list)


class ReferenceLibrary:
    def __init__(self, entries: list[LibraryEntry]):
        self.entries = entries
        if entries:
            self._matrix = np.stack([e.embedding for e in entries])
        else:
            self._matrix = np.zeros((0, 384), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def match(self, embedding: np.ndarray) -> tuple[LibraryEntry | None, float]:
        """Nearest-neighbour by cosine similarity (embeddings are
        pre-normalised, so this is a dot product)."""
        if len(self.entries) == 0:
            return None, 0.0
        sims = self._matrix @ embedding
        idx = int(np.argmax(sims))
        return self.entries[idx], float(sims[idx])

    @classmethod
    def build(cls, reference_dir: str | Path, cfg: ClassifyConfig) -> "ReferenceLibrary":
        """Scans reference_dir/<class_name>/*.png, augments each with the
        configured rotations (and mirror), embeds, and returns the index.

        An empty or missing reference_dir yields an empty library — every
        candidate then falls out as "unknown" (see classify/match.py) rather
        than raising, so the pipeline stays runnable before the library is
        populated (day-1 walking skeleton).
        """
        import cv2

        reference_dir = Path(reference_dir)
        if not reference_dir.exists():
            return cls([])

        embedder = Embedder(cfg.model)
        crops: list[np.ndarray] = []
        meta: list[tuple[str, int, bool, str]] = []
        terminals_per_crop: list[list[TerminalTemplate]] = []

        for class_dir in sorted(p for p in reference_dir.iterdir() if p.is_dir()):
            for img_path in sorted(class_dir.glob("*.png")):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                base_terminals = _load_terminals(img_path)
                variants = _augment(img_rgb, cfg.rotations, cfg.mirror, base_terminals)
                for rotation, mirrored, variant, variant_terminals in variants:
                    crops.append(variant)
                    meta.append((class_dir.name, rotation, mirrored, str(img_path)))
                    terminals_per_crop.append(variant_terminals)

        if not crops:
            return cls([])

        embeddings = embedder.embed(crops)
        entries = [
            LibraryEntry(class_name=cn, rotation=rot, mirrored=mir, source_path=sp, embedding=emb, terminals=term)
            for (cn, rot, mir, sp), emb, term in zip(meta, embeddings, terminals_per_crop)
        ]
        return cls(entries)


def _load_terminals(img_path: Path) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    """Reads the <name>.terminals.json sidecar (pixel coords in the base,
    unrotated image) if present. Missing sidecar -> no terminals; the
    symbol still classifies, terminals just fall back to "inferred" from
    wire contact points later (see types.Terminal.source)."""
    sidecar = img_path.with_suffix("").with_suffix(".terminals.json")
    if not sidecar.exists():
        return []
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    return [(entry["name"], tuple(entry["point"]), tuple(entry.get("body", entry["point"]))) for entry in data]


def _augment(
    img_rgb: np.ndarray,
    rotations: list[int],
    mirror: bool,
    base_terminals: list[tuple[str, tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, bool, np.ndarray, list[TerminalTemplate]]]:
    import cv2

    out = []
    rot_map = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    base_h, base_w = img_rgb.shape[:2]

    bases = [(img_rgb, False)]
    if mirror:
        bases.append((cv2.flip(img_rgb, 1), True))

    for base_img, is_mirrored in bases:
        if is_mirrored:
            mirrored_terminals = [
                (name, (base_w - 1 - tx, ty), (base_w - 1 - bx, by)) for name, (tx, ty), (bx, by) in base_terminals
            ]
        else:
            mirrored_terminals = base_terminals

        for rotation in rotations:
            code = rot_map.get(rotation)
            variant = base_img if code is None else cv2.rotate(base_img, code)
            vh, vw = variant.shape[:2]
            normalized: list[TerminalTemplate] = []
            for name, tip, body in mirrored_terminals:
                rtip = _rotate_point(*tip, base_w, base_h, rotation)
                rbody = _rotate_point(*body, base_w, base_h, rotation)
                normalized.append((name, (rtip[0] / vw, rtip[1] / vh), (rbody[0] / vw, rbody[1] / vh)))
            out.append((rotation, is_mirrored, variant, normalized))
    return out


def _rotate_point(x: float, y: float, w: int, h: int, rotation: int) -> tuple[float, float]:
    """Matches cv2.rotate's pixel mapping for 0/90/180/270 degree rotations
    of a (w, h) image."""
    if rotation == 0:
        return x, y
    if rotation == 90:  # cv2.ROTATE_90_CLOCKWISE
        return h - 1 - y, x
    if rotation == 180:
        return w - 1 - x, h - 1 - y
    if rotation == 270:  # cv2.ROTATE_90_COUNTERCLOCKWISE
        return y, w - 1 - x
    raise ValueError(f"unsupported rotation: {rotation}")
