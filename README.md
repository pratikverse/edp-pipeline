# Electrical Drawing Interpretation Pipeline

Reads a raster electrical drawing — detects symbols, classifies them, traces
wire connectivity, and emits a JSON netlist + a graph. Handles both
electronic circuit schematics and process & instrumentation diagrams (P&IDs)
through a domain‑pack architecture: the pipeline machinery is
drawing‑type‑agnostic, and everything specific to a drawing type lives in a
swappable pack (`edp/domains/electronic/`, `edp/domains/pid/`).

Full write‑up: **`docs/TECHNICAL_DOCUMENTATION.md`**. Design history and
measured experiments: `docs/08`–`docs/13`.

## Setup

```
conda env create -f environment.yml
conda activate edp
# GPU (recommended; training a detector on CPU is ~3 h):
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
bash scripts/bootstrap.sh          # warm the reference cache, regenerate P&ID symbols (~1 min)
```

The three shipped model artifacts (`data/models/`) and all source symbol
assets are in the repo, so the pipeline runs immediately after setup.
Retraining from scratch is documented in **Reproducing** below.

## Run

```
edp run data/validation/D4.png --out outputs/      # one drawing (auto-routes electronic/pid)
edp run data/validation/ --out outputs/            # a whole directory
edp eval --golden-dir data/golden --predicted-dir outputs/golden_prep
edp serve                                          # demo UI at http://localhost:8000
```

`edp run` writes `<id>.json` (the trimmed `id · type · coordinates ·
connections` schema) plus graph files. The demo UI shows four panels: the
input drawing, the detected‑component overlay + table, the JSON, and a
connectivity graph the browser builds from that JSON.

## What's measured (`edp eval`)

| drawing | domain | detection F1 | classification | connectivity F1 |
|---|---|---|---|---|
| D4 | electronic | 0.914 | 0.812 | 0.641 |
| D5 | electronic | 0.839 | 0.462 | — |
| D3 | P&ID | 0.741 | 0.550 | — |

Golden sets are in `data/golden/` (hand‑verified). D4 has hand‑traced
connections (`_connectivity_verified: true`); D3/D5 verify symbol type only.

## Reproducing the model artifacts

Not needed to run — the shipped weights are committed. To rebuild from
scratch:

```
# electronic detector (yolov8n, 17 classes) — ~30 min GPU
python scripts/build_reference_from_kicad.py                    # data/reference/ from data/kicad_raw/
python scripts/generate_synthetic_dataset.py --out-dir data/synth --n-train 800 --n-val 100
python scripts/generate_ladder_circuits.py  --out-dir data/synth --n-train 150 --append
python scripts/train_yolo.py --data data/synth/data.yaml --epochs 40 --name symbol_detector

# linear probe — ~5 min GPU
python scripts/train_linear_probe.py

# P&ID detector (yolov8n, 10 classes) — ~50 min GPU
python scripts/build_pid_reference.py
python scripts/generate_synthetic_dataset.py --reference-dir data/pid_reference \
    --out-dir data/synth_pid --n-train 900 --n-val 100 --size-min 40 --size-max 260
python scripts/train_yolo.py --data data/synth_pid/data.yaml --epochs 45 --name pid_synthetic_yolo

# optional: the class-agnostic real-data detector (docs/11 — measured, off by default)
python scripts/build_realdata_detector.py --epochs 30
```

## Layout

```
edp/
  pipeline.py  config.py  types.py  cli.py  eval.py
  preprocess.py  localize.py  realdetect.py  ocr.py  wires.py  emit.py  validate.py
  classify/    evidence · embedder · faiss_index · library · match · probe · text_prior · kicad_import
  domains/     base.py · electronic/ · pid/
  web/         server.py + static/
scripts/       dataset generation, training, dataset conversion
config/        default.yaml
data/          golden/ · kicad_raw/ · reference/ · pid_reference/ · validation/ · models/
tests/         47 tests — deterministic logic around the models
docs/          08–13: improvement plan, interview prep, codebase ref, experiments
```

Not tracked: `outputs/`, synthetic datasets, `data/models/realdata_symbol_detector.pt` (a
shelved experiment), `.env`.
