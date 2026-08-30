"""Trains a linear probe (logistic regression) on frozen DINOv2 embeddings
of domain-randomized synthetic symbol crops — the classification-evidence
architecture's Phase 7 experiment (docs/08_improvement_plan.md).

Tests whether "frozen embedding + learned decision boundary" beats "frozen
embedding + nearest reference" (the current classify/library.py path),
without touching that path: this trains an *additional*, independently
evaluable classifier. Wiring it into the fusion pipeline as a new evidence
source (if it measurably helps — see docs/08) is a separate step, done
only after `edp eval` says so.

Usage:
    python scripts/train_linear_probe.py --variants-per-source 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from domain_randomize import randomize_canvas, randomize_symbol  # noqa: E402

from edp.classify.embedder import Embedder  # noqa: E402
from edp.config import Config  # noqa: E402

REFERENCE_DIR = REPO_ROOT / "data" / "reference"
ROTATIONS = [0, 90, 180, 270]


def _rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(degrees)
    return img if code is None else cv2.rotate(img, code)


def _load_sources() -> dict[str, list[np.ndarray]]:
    sources: dict[str, list[np.ndarray]] = {}
    for class_dir in sorted(p for p in REFERENCE_DIR.iterdir() if p.is_dir()):
        images = []
        for img_path in sorted(class_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        if images:
            sources[class_dir.name] = images
    return sources


def generate_training_crops(
    sources: dict[str, list[np.ndarray]], variants_per_source: int, mirror: bool, seed: int
) -> tuple[list[np.ndarray], list[str]]:
    """Every (source image x rotation x mirror) combination, each repeated
    `variants_per_source` times with independent domain-randomization jitter
    — same rotation/mirror coverage as the existing reference library
    (classify/library.py's `_augment`) so the probe isn't weaker on
    orientation than nearest-neighbour already is, plus the noise/blur/
    stroke-width jitter nearest-neighbour never gets."""
    rng = random.Random(seed)
    crops: list[np.ndarray] = []
    labels: list[str] = []
    for class_name, images in sources.items():
        for base_img in images:
            bases = [base_img]
            if mirror:
                bases.append(cv2.flip(base_img, 1))
            for base in bases:
                for rotation in ROTATIONS:
                    variant_base = _rotate(base, rotation)
                    for _ in range(variants_per_source):
                        crop = randomize_symbol(variant_base.copy(), rng)
                        crop = randomize_canvas(crop, rng)
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crops.append(crop_rgb)
                        labels.append(class_name)
    return crops, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants-per-source", type=int, default=40)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="outputs/probe")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    sources = _load_sources()
    print(f"[probe] {sum(len(v) for v in sources.values())} source images across {len(sources)} classes")

    t0 = time.perf_counter()
    crops, labels = generate_training_crops(
        sources, args.variants_per_source, mirror=not args.no_mirror, seed=args.seed
    )
    print(f"[probe] generated {len(crops)} domain-randomized training crops in {time.perf_counter()-t0:.1f}s")

    embedder = Embedder(cfg.classify.model)
    t0 = time.perf_counter()
    # embed() runs its whole input as one forward-pass batch (fine for a
    # handful of candidate crops per drawing, the only caller until now --
    # see embedder.py) -- chunked here since a few thousand training crops
    # in one batch would blow past 4GB VRAM. Batch size chosen
    # conservatively for that budget, not measured/tuned against it.
    batch_size = 64
    embeddings = np.concatenate(
        [embedder.embed(crops[i : i + batch_size]) for i in range(0, len(crops), batch_size)], axis=0
    )
    print(f"[probe] embedded {len(crops)} crops ({embeddings.shape[1]}-dim) in {time.perf_counter()-t0:.1f}s")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )
    # sklearn >=1.7 always uses a multinomial loss for solvers that support
    # it (removed the now-redundant multi_class= kwarg entirely).
    clf = LogisticRegression(max_iter=2000, C=1.0)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    print(f"[probe] fit LogisticRegression in {time.perf_counter()-t0:.1f}s")

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"[probe] held-out synthetic accuracy: train={train_acc:.3f} val={val_acc:.3f}")
    print(
        "[probe] NOTE: this is accuracy on more of the same synthetic distribution the probe "
        "was trained on -- it is not evidence of real-drawing performance. Only `edp eval` "
        "against data/golden/ after wiring this into classify/match.py answers that question."
    )

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(clf, out_dir / "linear_probe.joblib")
    metadata = {
        "embedding_model": cfg.classify.model,
        "class_names": sorted(clf.classes_.tolist()),
        "variants_per_source": args.variants_per_source,
        "mirror": not args.no_mirror,
        "rotations": ROTATIONS,
        "seed": args.seed,
        "n_training_crops": len(crops),
        "train_accuracy": train_acc,
        "held_out_synthetic_val_accuracy": val_acc,
        "sklearn_model": "LogisticRegression",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[probe] saved -> {out_dir}/linear_probe.joblib, metadata.json")


if __name__ == "__main__":
    main()
