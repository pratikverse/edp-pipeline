"""Generates topologically realistic validation circuits: a closed
rectangular border, two horizontal rails (top/bottom), and several
vertical branches hanging between them — exactly the ladder-network
structure D4, D5, and the 50 downloaded sample circuits all share — built
from our own KiCad-rendered reference symbols.

Why this exists, separate from generate_synthetic_dataset.py: that
generator scatters symbols at random positions with simple point-to-point
wire stubs, which is fine for teaching a detector what a symbol *looks
like* but nothing like a real schematic's layout. It conflates two
different questions when used for validation: "does the model handle
realistic circuit topology" and "does it generalize to symbols it never
saw." The 50 downloaded real circuits answer the second question but not
cleanly the first, since they also use a different icon style than our
KiCad reference library (different transformer/battery symbols, a
potentiometer class we don't have at all) — a failure there could be
topology, style-mismatch, or both, hard to tell apart.

This generator isolates the topology question: same KiCad symbol style as
training (so a failure here can't be blamed on icon-style mismatch), but
real ladder-network layout (so a pass here means the architecture itself
handles realistic circuits, not just scattered icons).

Usage:
    python scripts/generate_ladder_circuits.py --n 60 --out-dir data/validation_kicad_topology
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
sys.path.insert(0, str(REPO_ROOT))

REFERENCE_DIR = REPO_ROOT / "data" / "reference"

TARGET_SIZE_RANGE = (28, 70)  # slightly tighter than the scatter generator's — branch
# components sit in a fixed-width lane, so keep them from dominating it
BRANCH_COUNT_RANGE = (5, 10)
COMPONENTS_PER_BRANCH_RANGE = (1, 2)  # mostly single, sometimes two in series
CANVAS_WIDTH_RANGE = (750, 1050)
CANVAS_HEIGHT_RANGE = (420, 560)
MARGIN = 50
JUNCTION_DOT_RADIUS = 3


@dataclass
class SourceSymbol:
    class_name: str
    image: np.ndarray


def load_sources() -> tuple[list[SourceSymbol], list[str]]:
    class_names = sorted(p.name for p in REFERENCE_DIR.iterdir() if p.is_dir())
    sources = []
    for class_name in class_names:
        for img_path in sorted((REFERENCE_DIR / class_name).glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                sources.append(SourceSymbol(class_name=class_name, image=img))
    return sources, class_names


def _resize_to_target(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    target_long = rng.uniform(*TARGET_SIZE_RANGE)
    scale = target_long / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _paste(canvas: np.ndarray, symbol_img: np.ndarray, x0: int, y0: int) -> None:
    h, w = symbol_img.shape[:2]
    mask = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY) < 250
    region = canvas[y0 : y0 + h, x0 : x0 + w]
    region[mask] = symbol_img[mask]


def generate_one(sources: list[SourceSymbol], class_names: list[str], rng: random.Random):
    w = rng.randint(*CANVAS_WIDTH_RANGE)
    h = rng.randint(*CANVAS_HEIGHT_RANGE)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    top_rail_y = MARGIN
    bottom_rail_y = h - MARGIN
    left_x = MARGIN
    right_x = w - MARGIN

    n_branches = rng.randint(*BRANCH_COUNT_RANGE)
    branch_xs = np.linspace(left_x + 30, right_x - 30, n_branches).astype(int)

    labels: list[tuple[int, float, float, float, float]] = []

    # Closed rectangular border (top/bottom rails are part of this frame,
    # exactly like D4/D5's outer loop).
    cv2.rectangle(canvas, (left_x, top_rail_y), (right_x, bottom_rail_y), (0, 0, 0), 1)

    for bx in branch_xs:
        bx = int(bx)
        n_components = rng.randint(*COMPONENTS_PER_BRANCH_RANGE)
        variants = []
        for _ in range(n_components):
            src = rng.choice(sources)
            img = _resize_to_target(src.image, rng)
            variants.append((src.class_name, img))

        total_h = sum(v[1].shape[0] for v in variants)
        gap = 24
        total_h_with_gaps = total_h + gap * (n_components + 1)
        available_h = (bottom_rail_y - top_rail_y) - 20
        if total_h_with_gaps > available_h:
            # Too tall for this branch (rare, small canvases) — skip placing
            # components here but still draw the branch wire, matching a
            # plain unbroken connection between rails.
            cv2.line(canvas, (bx, top_rail_y), (bx, bottom_rail_y), (0, 0, 0), 1, cv2.LINE_AA)
            continue

        start_y = top_rail_y + (available_h - total_h_with_gaps) // 2 + 10
        cursor_y = start_y

        # Wire stub from top rail down to the first component.
        cv2.line(canvas, (bx, top_rail_y), (bx, cursor_y), (0, 0, 0), 1, cv2.LINE_AA)

        for class_name, img in variants:
            ih, iw = img.shape[:2]
            x0 = bx - iw // 2
            y0 = cursor_y
            _paste(canvas, img, x0, y0)

            class_id = class_names.index(class_name)
            xc, yc = (x0 + iw / 2) / w, (y0 + ih / 2) / h
            bw, bh = iw / w, ih / h
            labels.append((class_id, xc, yc, bw, bh))

            cursor_y = y0 + ih + gap
            cv2.line(canvas, (bx, y0 + ih), (bx, min(cursor_y, bottom_rail_y)), (0, 0, 0), 1, cv2.LINE_AA)

        # Final stub down to the bottom rail.
        cv2.line(canvas, (bx, cursor_y - gap), (bx, bottom_rail_y), (0, 0, 0), 1, cv2.LINE_AA)

        # Junction dots where the branch meets each rail — matches the
        # real drawing convention (docs/01 stage 6) and D4/D5's own style.
        cv2.circle(canvas, (bx, top_rail_y), JUNCTION_DOT_RADIUS, (0, 0, 0), -1)
        cv2.circle(canvas, (bx, bottom_rail_y), JUNCTION_DOT_RADIUS, (0, 0, 0), -1)

    return canvas, labels


def write_split(sources, class_names, n_images: int, out_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        canvas, labels = generate_one(sources, class_names, rng)
        stem = f"ladder_{i:04d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), canvas)
        with open(lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            for class_id, xc, yc, bw, bh in labels:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[ladder] wrote {n_images} images to {img_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out-dir", type=str, default="data/validation_kicad_topology")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    sources, class_names = load_sources()
    print(f"[ladder] {len(sources)} source symbols across {len(class_names)} classes")

    write_split(sources, class_names, args.n, out_dir, seed=args.seed)

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.as_posix()}",
                # ultralytics requires both keys even for a validation-only set;
                # 'train' is never actually used since we only call model.val().
                "train: images",
                "val: images",
                f"names: {class_names}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[ladder] data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
