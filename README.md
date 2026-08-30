# Electrical Drawing Interpretation Pipeline

Detects symbols, traces wire connectivity, and emits a JSON netlist + graph
from raster electrical drawings.

## Setup

```
conda env create -f environment.yml
conda activate edp
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124  # GPU
pip install -e .
```

## Run

```
edp run data/validation/D4.png --out outputs/
edp eval --golden-dir data/golden --predicted-dir outputs/golden_prep
edp serve
```

Not tracked (kept on disk): `data/`, model weights, `.env`.
