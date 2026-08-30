#!/usr/bin/env bash
# One-time setup for a fresh checkout. Assumes the conda env `edp` is
# already created and activated (see README) and `pip install -e .` has
# run. Idempotent — safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[bootstrap] checking shipped model artifacts ..."
missing=0
for f in data/models/synthetic_yolo_17class.pt data/models/pid_synthetic_yolo.pt data/models/linear_probe.joblib; do
  if [ ! -f "$f" ]; then echo "  MISSING: $f"; missing=1; fi
done
if [ "$missing" = 1 ]; then
  echo "[bootstrap] some weights are missing — see 'Reproducing' in README.md to retrain."
fi

echo "[bootstrap] regenerating the procedural P&ID reference symbols ..."
python scripts/build_pid_reference.py

echo "[bootstrap] warming the DINOv2 reference-embedding cache (~1 min, downloads DINOv2 once) ..."
python -m edp.cli build-library data/reference

echo
echo "[bootstrap] done. Try:"
echo "    edp run data/validation/D4.png --out outputs/"
echo "    edp eval --golden-dir data/golden --predicted-dir outputs/golden_prep"
echo "    edp serve      # http://localhost:8000"
