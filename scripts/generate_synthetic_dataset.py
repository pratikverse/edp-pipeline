"""Generates a synthetic YOLO training set by compositing our existing
KiCad-rendered reference symbols onto blank schematic-like canvases, with
auto-generated bounding-box labels.

Why synthetic composition rather than a found dataset (see docs/05): a
schematic symbol is a man-made icon with one canonical look, not a
physical object with photographic variation — KiCad's own library already
IS the definitive "photo" for this domain, at higher fidelity and zero
licensing risk than any scraped icon set. What YOLO actually needs is
*scenes* with several symbols placed together and labelled boxes, which no
found dataset hands you regardless of source — so the real work is this
composition step either way. Labels are self-generated, not manually
annotated, so this does not reintroduce hand-labeled training data.

Usage:
    python scripts/generate_synthetic_dataset.py --n-train 800 --n-val 100
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REFERENCE_DIR = REPO_ROOT / "data" / "reference"
OUT_DIR = REPO_ROOT / "data" / "synth"

# Real symbols in D4/D5 run roughly 25-90px on a side; our KiCad renders are
# 20px/mm and much larger (up to 430px) since that scale was chosen for
# clean DINOv2 embedding, not matched to real drawing scale. Composing at
# the real distribution is what makes the trained detector transfer.
TARGET_SIZE_RANGE = (24, 85)
CANVAS_SIZE_RANGE = (500, 900)
SYMBOLS_PER_CANVAS_RANGE = (6, 16)
ROTATIONS = [0, 90, 180, 270]


@dataclass
class SourceSymbol:
    class_name: str
    image: np.ndarray  # BGR, white background


def load_sources() -> tuple[list[SourceSymbol], list[str]]:
    class_names = sorted(p.name for p in REFERENCE_DIR.iterdir() if p.is_dir())
    sources = []
    for class_name in class_names:
        for img_path in sorted((REFERENCE_DIR / class_name).glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                sources.append(SourceSymbol(class_name=class_name, image=img))
    return sources, class_names


def _rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(degrees)
    return img if code is None else cv2.rotate(img, code)


def _resize_to_target(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    target_long = random.uniform(*TARGET_SIZE_RANGE)
    scale = target_long / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _draw_manhattan_wire(canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    """A simple orthogonal L-route between two points — cheap wire-clutter
    realism so the detector sees symbols in context, not on a blank page,
    since real candidate crops always include nearby wire fragments."""
    x1, y1 = p1
    x2, y2 = p2
    mid = (x2, y1) if random.random() < 0.5 else (x1, y2)
    cv2.line(canvas, (x1, y1), mid, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(canvas, mid, (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)


def _draw_text_clutter(canvas: np.ndarray, rng: random.Random) -> None:
    """Short dash clusters mimicking OCR labels — unlabelled negatives, so
    the detector learns not to fire on text the way the classical
    density-based localizer did (see docs/01 known limitations)."""
    h, w = canvas.shape[:2]
    x, y = rng.randint(10, w - 30), rng.randint(10, h - 15)
    for i in range(rng.randint(2, 4)):
        dash_w = rng.randint(4, 10)
        cv2.line(canvas, (x, y), (x + dash_w, y), (0, 0, 0), 1, cv2.LINE_AA)
        x += dash_w + rng.randint(2, 4)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def generate_one(sources: list[SourceSymbol], class_names: list[str], rng: random.Random):
    h = rng.randint(*CANVAS_SIZE_RANGE)
    w = rng.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Border rail, like the frame most real schematics are drawn inside.
    cv2.rectangle(canvas, (5, 5), (w - 5, h - 5), (0, 0, 0), 1)

    n_symbols = rng.randint(*SYMBOLS_PER_CANVAS_RANGE)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_centers: list[tuple[int, int]] = []
    labels: list[tuple[int, float, float, float, float]] = []  # class_id, xc, yc, bw, bh (normalized)

    for _ in range(n_symbols):
        src = rng.choice(sources)
        variant = _resize_to_target(src.image)
        variant = _rotate(variant, rng.choice(ROTATIONS))
        sh, sw = variant.shape[:2]
        if sh >= h - 20 or sw >= w - 20:
            continue

        placed = False
        for _attempt in range(15):
            x0 = rng.randint(15, w - sw - 15)
            y0 = rng.randint(15, h - sh - 15)
            box = (x0, y0, x0 + sw, y0 + sh)
            if all(_iou(box, existing) < 0.05 for existing in placed_boxes):
                placed = True
                break
        if not placed:
            continue

        mask = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) < 250
        region = canvas[y0 : y0 + sh, x0 : x0 + sw]
        region[mask] = variant[mask]

        placed_boxes.append(box)
        placed_centers.append((x0 + sw // 2, y0 + sh // 2))

        class_id = class_names.index(src.class_name)
        xc, yc = (x0 + sw / 2) / w, (y0 + sh / 2) / h
        bw, bh = sw / w, sh / h
        labels.append((class_id, xc, yc, bw, bh))

    # Wire clutter between some nearby symbol pairs.
    for i in range(len(placed_centers)):
        if rng.random() < 0.5 and i > 0:
            j = rng.randint(0, i - 1)
            _draw_manhattan_wire(canvas, placed_centers[i], placed_centers[j])

    for _ in range(rng.randint(2, 6)):
        _draw_text_clutter(canvas, rng)

    return canvas, labels


def write_split(sources, class_names, n_images: int, split: str, seed: int) -> None:
    rng = random.Random(seed)
    img_dir = OUT_DIR / "images" / split
    lbl_dir = OUT_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        canvas, labels = generate_one(sources, class_names, rng)
        stem = f"{split}_{i:05d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), canvas)
        with open(lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            for class_id, xc, yc, bw, bh in labels:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[synth] wrote {n_images} images to {img_dir}")


def write_data_yaml(class_names: list[str]) -> Path:
    yaml_path = OUT_DIR / "data.yaml"
    lines = [
        f"path: {OUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"names: {class_names}",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sources, class_names = load_sources()
    print(f"[synth] {len(sources)} source symbol images across {len(class_names)} classes: {class_names}")

    write_split(sources, class_names, args.n_train, "train", seed=args.seed)
    write_split(sources, class_names, args.n_val, "val", seed=args.seed + 1)
    yaml_path = write_data_yaml(class_names)
    print(f"[synth] data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
