"""Class-agnostic real-data symbol detector (docs/11).

Remaps the nadim-ahmed/circuit-component-detection dataset (17 component
classes) down to a single "symbol" class and trains a small YOLO detector
on it, for use as the class-agnostic second localizer in edp/realdetect.py
(provider: local). The `Wire_Overlap` class is dropped, not remapped — a
wire crossing is not a symbol and we don't want the detector proposing
boxes on junctions.

    python scripts/build_realdata_detector.py --epochs 60

Input:  data/roboflow_raw/nadim_v21/   (downloaded via the roboflow API)
Output: data/models/realdata_symbol_detector.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "roboflow_raw" / "nadim_v21"
WORK = REPO / "data" / "roboflow_raw" / "nadim_v21_agnostic"
OUT = REPO / "data" / "models" / "realdata_symbol_detector.pt"

DROP_CLASS_NAMES = {"Wire_Overlap"}


def _prep() -> Path:
    names = yaml.safe_load((SRC / "data.yaml").read_text())["names"]
    drop_ids = {i for i, n in enumerate(names) if n in DROP_CLASS_NAMES}
    print(f"[prep] dropping class ids {drop_ids} ({DROP_CLASS_NAMES}); all others -> 0")

    if WORK.exists():
        shutil.rmtree(WORK)
    for split in ("train", "valid"):
        (WORK / split / "images").mkdir(parents=True, exist_ok=True)
        (WORK / split / "labels").mkdir(parents=True, exist_ok=True)
        img_dir = SRC / split / "images"
        if not img_dir.exists():
            continue
        kept = 0
        for lbl in (SRC / split / "labels").glob("*.txt"):
            out_lines = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if int(parts[0]) in drop_ids:
                    continue
                out_lines.append("0 " + " ".join(parts[1:]))
            (WORK / split / "labels" / lbl.name).write_text("\n".join(out_lines) + "\n")
            stem = lbl.stem
            for ext in (".jpg", ".png", ".jpeg"):
                src_img = img_dir / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy(src_img, WORK / split / "images" / src_img.name)
                    kept += 1
                    break
        print(f"[prep] {split}: {kept} images")

    data_yaml = WORK / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(WORK),
                "train": "train/images",
                "val": "valid/images",
                "nc": 1,
                "names": ["symbol"],
            }
        )
    )
    return data_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)  # RTX 3050 4GB
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    data_yaml = _prep()
    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    print(f"[train] {args.model} on {data_yaml} — {args.epochs} epochs, device={device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(REPO / "outputs" / "yolo_runs"),
        name="realdata_symbol_detector",
        exist_ok=True,
        single_cls=True,
    )
    best = REPO / "outputs" / "yolo_runs" / "realdata_symbol_detector" / "weights" / "best.pt"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OUT)
    print(f"[train] saved -> {OUT}")


if __name__ == "__main__":
    main()
