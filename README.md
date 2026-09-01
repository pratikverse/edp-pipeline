# Electrical Drawing Interpretation Pipeline

Reads a raster electrical drawing — detects symbols, classifies them, traces
wire connectivity, and emits a JSON netlist + a graph. Handles both
electronic circuit schematics and process & instrumentation diagrams (P&IDs)
through a domain‑pack architecture: the pipeline machinery is
drawing‑type‑agnostic, and everything specific to a drawing type lives in a
swappable pack (`edp/domains/electronic/`, `edp/domains/pid/`).

## Setup

```
conda env create -f environment.yml
conda activate edp
# GPU (optional; CPU works, just slower):
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .          # optional — or use `python -m edp.cli ...` below
```

The shipped model artifacts (`data/models/`) and all source symbol assets are
in the repo, so the pipeline runs immediately after setup. The reference
embedding library builds itself on first use and caches to
`data/reference/index.npz` (first run is ~30–60 s slower).

OCR needs Tesseract; `environment.yml` pulls it in. If you hit
`TesseractNotFoundError`, run `conda install -c conda-forge tesseract`.

## Run

```
edp run data/validation/D4.png --out outputs/      # one drawing (auto-routes electronic/pid)
edp run data/validation/ --out outputs/            # a whole directory
edp eval --golden-dir data/golden --predicted-dir outputs/golden_prep
edp serve                                          # demo UI at http://localhost:8000
```

Without the editable install, use `python -m edp.cli` in place of `edp`.

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

## Layout

```
edp/
  pipeline.py  config.py  types.py  cli.py  eval.py
  preprocess.py  localize.py  realdetect.py  ocr.py  wires.py  emit.py  validate.py
  classify/    evidence · embedder · faiss_index · library · match · probe · text_prior · kicad_import
  domains/     base.py · electronic/ · pid/
  web/         server.py + static/
config/        default.yaml
data/          golden/ · kicad_raw/ · reference/ · pid_reference/ · validation/ · models/
```

The three shipped detectors/probe under `data/models/` are prebuilt; the
training and dataset-generation code is not included in this repo.

Not tracked: `outputs/`, synthetic datasets,
`data/models/realdata_symbol_detector.pt` (a shelved experiment), `.env`.
