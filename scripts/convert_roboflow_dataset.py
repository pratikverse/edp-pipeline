"""Converts the "Circuit Recognition" dataset from Roboflow Universe
(https://universe.roboflow.com/rp-project/circuit-recognition, MIT
licensed) into this project's YOLO label format, for mixing into
data/synth_mixed alongside the KiCad-sourced synthetic composites.

Only 4 of its 8 classes have a real, visually-verified equivalent in our
21-class taxonomy -- checked by cropping and looking at actual sample
images before trusting any mapping, not assumed from class names alone
(see the conversation record):

    r  -> Resistor      (ANSI zigzag, matches our own convention)
    c  -> Capacitor      (flat parallel plates, matches)
    l  -> Inductor        (drawn coil, matches)
    v  -> Battery          (long/short plate pair, matches our Battery
                            convention closely -- NOT a generic "circle"
                            voltage-source symbol as the class name alone
                            might suggest)

Deliberately EXCLUDED, and why:
    acv -> AC voltage source, drawn as a circle -- no equivalent class,
           and including it under any of ours would just teach the
           detector that circle-shaped things are that wrong class
    i   -> current source, also a circle+arrow -- same problem, and
           visually close to how our own BJT/MOSFET symbols use a circle,
           so wrongly labelling it under either would actively hurt
    arr -> a wire/current-direction arrow annotation, not a component
    l-  -> an unclear inductor variant/typo class, skipped conservatively
           rather than guessed at

An image keeps only its boxes for the 4 mapped classes; boxes for
excluded classes are dropped (not converted to "background") from that
image's label file -- same principle as our own generators' unlabelled
wire/text clutter: present in the image, just not something the detector
is asked to learn from. Images left with zero boxes after filtering are
skipped entirely.

Usage:
    python scripts/convert_roboflow_dataset.py \\
        --source data/roboflow_raw/extracted \\
        --out-dir data/synth_mixed --prefix rf
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Source class index (from the downloaded data.yaml's `names` list) -> our
# class name. Anything not in this dict is dropped.
SOURCE_CLASS_NAMES = ["acv", "arr", "c", "i", "l", "l-", "r", "v"]
CLASS_MAP = {
    "c": "Capacitor",
    "l": "Inductor",
    "r": "Resistor",
    "v": "Battery",
}


def _our_class_names() -> list[str]:
    """Same `sorted(class_dir names)` convention generate_synthetic_dataset.py
    uses, so label indices line up with whatever the rest of the pipeline
    assigns when it enumerates data/reference/ at training time."""
    reference_dir = REPO_ROOT / "data" / "reference"
    return sorted(p.name for p in reference_dir.iterdir() if p.is_dir())


def convert_split(
    source_dir: Path, split: str, out_dir: Path, prefix: str, class_names: list[str], max_images: int | None, rng: random.Random
) -> tuple[int, int]:
    img_dir = source_dir / split / "images"
    lbl_dir = source_dir / split / "labels"
    if not img_dir.exists():
        return 0, 0

    out_img_dir = out_dir / "images" / "train"
    out_lbl_dir = out_dir / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(lbl_dir.glob("*.txt"))
    if max_images is not None and len(label_paths) > max_images:
        # Random subsample, not first-N: Roboflow's own export ordering may
        # cluster near-duplicate augmented variants of the same source image
        # together (their "8 versions" apply augmentation multipliers -- see
        # the conversation record), so a first-N slice risks low diversity
        # relative to a random draw across the whole pool.
        label_paths = rng.sample(label_paths, max_images)

    kept_images = 0
    kept_boxes = 0
    for label_path in label_paths:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        remapped = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            source_class = SOURCE_CLASS_NAMES[int(parts[0])]
            our_class = CLASS_MAP.get(source_class)
            if our_class is None:
                continue
            new_id = class_names.index(our_class)
            remapped.append(f"{new_id} {' '.join(parts[1:])}")
        if not remapped:
            continue

        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = img_dir / (label_path.stem + ext)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        stem = f"{prefix}_{split}_{label_path.stem}"
        shutil.copy(image_path, out_img_dir / f"{stem}{image_path.suffix}")
        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(remapped) + "\n", encoding="utf-8")
        kept_images += 1
        kept_boxes += len(remapped)

    return kept_images, kept_boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/roboflow_raw/extracted")
    parser.add_argument("--out-dir", default="data/synth_mixed")
    parser.add_argument("--prefix", default="rf")
    parser.add_argument(
        "--max-images",
        type=int,
        default=300,
        help="Cap on how many source images to convert (per split), subsampled randomly. "
        "Default 300 keeps this dataset roughly proportionate to the ~950-image KiCad-based "
        "synthetic set it gets mixed into -- the raw download has 3352 usable images across "
        "only 4 of our 21 classes, and using all of them would swamp the other 17 classes' "
        "representation and skew training toward this one dataset's specific drawing style.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_dir = REPO_ROOT / args.source
    out_dir = REPO_ROOT / args.out_dir
    class_names = _our_class_names()
    rng = random.Random(args.seed)
    print(f"[rf-convert] target class list ({len(class_names)}): {class_names}")
    print(f"[rf-convert] mapping: {CLASS_MAP}")
    print(f"[rf-convert] max_images per split: {args.max_images}")

    total_images = 0
    total_boxes = 0
    for split in ("train", "valid", "test"):
        n_img, n_box = convert_split(source_dir, split, out_dir, args.prefix, class_names, args.max_images, rng)
        print(f"[rf-convert] {split}: {n_img} images kept, {n_box} boxes kept")
        total_images += n_img
        total_boxes += n_box

    print(f"[rf-convert] TOTAL: {total_images} images, {total_boxes} boxes -> {out_dir}/images/train, labels/train")


if __name__ == "__main__":
    main()
