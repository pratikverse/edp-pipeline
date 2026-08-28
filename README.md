# Electrical Drawing Interpretation Pipeline

See `docs/` for architecture, model-selection rationale, JSON schema,
implementation plan, and open questions/assumptions.

## Setup

```bash
conda env create -f environment.yml
conda activate edp
pip install -e .
```

## Usage

```bash
# run on a single drawing
edp run data/raw/D5.png --out outputs/

# run on every PNG in a directory
edp run data/raw/ --out outputs/

# rebuild the reference embedding library after adding symbols to
# data/reference/<class_name>/*.png
edp build-library data/reference/ --out data/reference/index.npz

# demo frontend: upload an image, get JSON + graph side by side
edp serve --port 8000
```

## Tests

```bash
pytest tests/
```

## Layout

See `docs/07_project_layout.md` for the full rationale. Summary:

```
src/edp/           pipeline package (preprocess, localize, classify, text,
                    wires, validate, emit, web)
config/             thresholds (default.yaml) — never hardcoded in code
data/raw/           input drawings (D4.png, D5.png)
data/reference/     reference symbol crops, one directory per class
outputs/            per-run JSON, graphs, debug overlays
tests/              synthetic unit tests + JSON schema contract tests
```
