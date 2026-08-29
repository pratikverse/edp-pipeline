"""Trains a YOLOv8n detector on the synthetic dataset from
generate_synthetic_dataset.py / generate_ladder_circuits.py. Uses GPU
(CUDA) automatically when available — the first two training runs in this
project's history were CPU-only (~3.2-3.5h for 40 epochs on ~600-700
images); a CUDA-enabled torch install cuts this dramatically. See
docs/02_model_selection_rationale.md.

Usage:
    python scripts/train_yolo.py --epochs 40
    python scripts/train_yolo.py --data data/synth_dense/data.yaml --name symbol_detector_dense
    python scripts/train_yolo.py --device cpu   # force CPU even if a GPU is available
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Same Windows conda/pip OpenMP runtime collision documented in
# edp/classify/embedder.py — this script is a separate process that never
# imports that module, so the guard has to be set here too, before torch
# (pulled in by ultralytics) loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--data", type=str, default="data/synth/data.yaml")
    parser.add_argument("--name", type=str, default="symbol_detector")
    parser.add_argument("--device", type=str, default=None, help="'0' for first GPU, 'cpu' to force CPU; auto-detects if omitted")
    args = parser.parse_args()

    import torch

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device != "cpu" else ""))

    data_yaml = REPO_ROOT / args.data
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(REPO_ROOT / "outputs" / "yolo_runs"),
        name=args.name,
        patience=15,
        verbose=True,
    )


if __name__ == "__main__":
    main()
