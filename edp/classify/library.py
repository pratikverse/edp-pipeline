"""Reference embedding library: build from data/reference/<class>/*.png,
rotation/mirror-augmented, embedded once and cached to disk. See
docs/02_model_selection_rationale.md (rotation handling, style variants)
and docs/07_project_layout.md (`edp build-library`).

`build()` auto-caches the expensive part (the DINOv2 forward pass) to
`<reference_dir>/index.npz`, keyed by a signature over every source file's
path/mtime/size plus the embedding model name and rotation/mirror config
— change any of those and the cache misses and rebuilds itself
automatically, so there's no separate "did I remember to rebuild the
index" step. Image loading and rotation/mirror augmentation still run on
every call regardless (decoding and rotating a few dozen small PNGs is
milliseconds; re-embedding all of them on every `edp run` invocation
measured 60-85s once the library grew past ~250 entries — see docs/08
section 1.3's "cost found along the way" note — which is what this cache
actually avoids).

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
from .faiss_index import FaissLibraryIndex

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
            self._matrix = np.zeros((0, 768), dtype=np.float32)
        self._index = FaissLibraryIndex(self._matrix)

    def __len__(self) -> int:
        return len(self.entries)

    def match(self, embedding: np.ndarray) -> tuple[LibraryEntry | None, float]:
        """Nearest-neighbour by cosine similarity, via the FAISS index
        (embeddings are pre-normalised, so inner product = cosine sim)."""
        if len(self.entries) == 0:
            return None, 0.0
        indices, scores = self._index.search(embedding, k=1)
        if len(indices) == 0:
            return None, 0.0
        return self.entries[int(indices[0])], float(scores[0])

    def match_topk(self, embedding: np.ndarray, k: int = 3, search_k: int = 20) -> list[tuple[str, float, "LibraryEntry"]]:
        """Top-k *distinct classes* by best cosine similarity, for the
        top-k-aware fusion in classify/match.py. `search_k` over-fetches
        raw neighbours before collapsing to classes, since the library
        holds many rotation/mirror-augmented entries per class — without
        over-fetching, the top few raw hits can all be the same class's
        augmented variants and starve genuinely distinct candidates out of
        the top-k entirely."""
        if len(self.entries) == 0:
            return []
        indices, scores = self._index.search(embedding, k=min(search_k, len(self.entries)))
        best_per_class: dict[str, tuple[float, LibraryEntry]] = {}
        for idx, score in zip(indices, scores):
            entry = self.entries[int(idx)]
            current = best_per_class.get(entry.class_name)
            if current is None or score > current[0]:
                best_per_class[entry.class_name] = (float(score), entry)
        ranked = sorted(best_per_class.items(), key=lambda kv: kv[1][0], reverse=True)
        return [(class_name, score, entry) for class_name, (score, entry) in ranked[:k]]

    @classmethod
    def build(
        cls, reference_dir: str | Path, cfg: ClassifyConfig, cache_path: str | Path | None = None
    ) -> "ReferenceLibrary":
        """Scans reference_dir/<class_name>/*.png, augments each with the
        configured rotations (and mirror), embeds (from cache when valid —
        see module docstring), and returns the index.

        An empty or missing reference_dir yields an empty library — every
        candidate then falls out as "unknown" (see classify/match.py) rather
        than raising, so the pipeline stays runnable before the library is
        populated (day-1 walking skeleton).
        """
        import cv2

        reference_dir = Path(reference_dir)
        if not reference_dir.exists():
            return cls([])
        cache_path = Path(cache_path) if cache_path is not None else reference_dir / "index.npz"

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

        signature = _cache_signature(reference_dir, cfg)
        embeddings = _load_cached_embeddings(cache_path, signature, meta)
        if embeddings is None:
            embedder = Embedder(cfg.model)
            embeddings = embedder.embed(crops)
            _save_cache(cache_path, embeddings, meta, signature)

        entries = [
            LibraryEntry(class_name=cn, rotation=rot, mirrored=mir, source_path=sp, embedding=emb, terminals=term)
            for (cn, rot, mir, sp), emb, term in zip(meta, embeddings, terminals_per_crop)
        ]
        return cls(entries)


def _cache_signature(reference_dir: Path, cfg: ClassifyConfig) -> str:
    """Hashes everything that would change the embeddings if it changed:
    every source file's path/mtime/size under reference_dir (covers add,
    remove, edit, or replace a reference image or its terminals sidecar),
    plus the embedding model name and rotation/mirror settings (changing
    either changes what gets fed to the embedder). Anything else in
    ClassifyConfig (thresholds, paths unrelated to the library) doesn't
    affect what's embedded, so isn't part of the signature — a change
    there shouldn't force a needless re-embed."""
    import hashlib

    files = sorted(
        p for p in reference_dir.rglob("*") if p.is_file() and p.suffix in (".png", ".json") and p.name != "index.npz"
    )
    parts = [f"model={cfg.model}", f"rotations={sorted(cfg.rotations)}", f"mirror={cfg.mirror}"]
    for p in files:
        stat = p.stat()
        parts.append(f"{p.relative_to(reference_dir).as_posix()}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _load_cached_embeddings(
    cache_path: Path, signature: str, expected_meta: list[tuple[str, int, bool, str]]
) -> np.ndarray | None:
    """Returns cached embeddings only if the signature matches (nothing in
    the reference dir or embedding config changed) AND the cached entry
    order/identity matches exactly what this build pass just enumerated —
    the second check is redundant with the signature in practice (a
    changed file set changes the signature too) but is cheap and makes the
    "never silently serve embeddings for the wrong crop" guarantee
    explicit rather than assumed."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if "signature" not in data.files or str(data["signature"]) != signature:
            return None
        cached_meta = list(
            zip(
                [str(x) for x in data["class_names"]],
                [int(x) for x in data["rotations"]],
                [bool(x) for x in data["mirrored"]],
                [str(x) for x in data["source_paths"]],
            )
        )
        if cached_meta != expected_meta:
            return None
        return data["embeddings"]
    except Exception:
        # Any read/format problem (corrupt file, older cache schema, etc.)
        # -- fall back to a full rebuild rather than raising, same
        # "degrade gracefully" precedent as the missing-reference-dir case.
        return None


def _save_cache(cache_path: Path, embeddings: np.ndarray, meta: list[tuple[str, int, bool, str]], signature: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embeddings,
        class_names=[m[0] for m in meta],
        rotations=[m[1] for m in meta],
        mirrored=[m[2] for m in meta],
        source_paths=[m[3] for m in meta],
        signature=signature,
    )


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
