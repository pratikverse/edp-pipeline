"""Trains a YOLOv8n detector on the synthetic dataset from
generate_synthetic_dataset.py. CPU-only environment (no CUDA), so this
uses the smallest model and a modest epoch count to stay tractable —
see docs/02_model_selection_rationale.md for the rationale entry once
this is folded into the pipeline.

Usage:
    python scripts/train_yolo.py --epochs 40
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
    args = parser.parse_args()

    data_yaml = REPO_ROOT / "data" / "synth" / "data.yaml"
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(REPO_ROOT / "outputs" / "yolo_runs"),
        name="symbol_detector",
        patience=15,
        verbose=True,
    )


if __name__ == "__main__":
    main()
