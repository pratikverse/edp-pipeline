"""Runs the full edp pipeline (YOLO localizer + DINOv2/FAISS classify) over
every image in data/validation/ — real circuit diagrams (D4, D5, plus the
50 downloaded sample circuits), not synthetic composites. There is no
ground-truth bounding-box labeling for these, so this produces overlays
for manual visual audit, the same method used to evaluate D4/D5 throughout
docs/02 and docs/05 — it does not compute an automated mAP.

Usage:
    python scripts/validate_on_real.py
    python scripts/validate_on_real.py --limit 10   # spot check a subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edp.config import Config  # noqa: E402
from edp.emit.json_out import to_json_dict  # noqa: E402
from edp.pipeline import run  # noqa: E402


def render_overlay(image_path: Path, symbols: list[dict], out_path: Path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    for s in symbols:
        x0, y0, x1, y1 = s["coordinates"]
        color = (0, 140, 255) if s["type"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        cv2.putText(img, f"{s['id']} {s['type'][:8]}", (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", default="data/validation")
    parser.add_argument("--out-dir", default="outputs/real_validation")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    val_dir = REPO_ROOT / args.validation_dir
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(val_dir.glob("*.png"))
    if args.limit:
        images = images[: args.limit]

    cfg = Config.load()
    print(f"[validate] {len(images)} real circuit images, using YOLO weights: {cfg.localize.yolo_weights}")

    summary = []
    for image_path in images:
        result, timing = run(image_path, cfg)
        json_dict = to_json_dict(result)
        n_symbols = len(json_dict["symbols"])
        n_connected = sum(1 for s in json_dict["symbols"] if s["connections"])
        n_unknown = sum(1 for s in json_dict["symbols"] if s["type"] == "Unknown")

        overlay_path = out_dir / f"{image_path.stem}_overlay.png"
        render_overlay(image_path, json_dict["symbols"], overlay_path)

        summary.append((image_path.name, n_symbols, n_connected, n_unknown))
        print(f"[validate] {image_path.name}: {n_symbols} symbols, {n_connected} connected, {n_unknown} unknown")

    print("\n[validate] summary:")
    print(f"{'file':<20} {'symbols':>8} {'connected':>10} {'unknown':>8}")
    for name, n_sym, n_conn, n_unk in summary:
        print(f"{name:<20} {n_sym:>8} {n_conn:>10} {n_unk:>8}")

    total_symbols = sum(s[1] for s in summary)
    total_connected = sum(s[2] for s in summary)
    total_unknown = sum(s[3] for s in summary)
    print(f"\n[validate] totals: {total_symbols} symbols across {len(images)} drawings, "
          f"{total_connected} connected ({100*total_connected/max(total_symbols,1):.0f}%), "
          f"{total_unknown} unknown ({100*total_unknown/max(total_symbols,1):.0f}%)")
    print(f"[validate] overlays written to {out_dir}")


if __name__ == "__main__":
    main()
