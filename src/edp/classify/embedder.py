"""DINOv2 embedding wrapper. Lazily loaded — importing this module must not
require a GPU or network access; the weights are only fetched/loaded on
first call to `embed()`. Keeps `edp run` usable with a stubbed/empty
reference library (day-1 walking skeleton) without pulling model weights."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# This Windows conda env mixes conda-forge's MKL-linked OpenCV/NumPy stack
# (libiomp5md.dll) with pip's official CPU torch wheel (bundles libomp.dll).
# Both are OpenMP runtimes; loading both in one process trips libomp's
# duplicate-runtime guard (OMP: Error #15) before torch even finishes
# importing. The clean fix (conda's `nomkl` to drop MKL entirely) forces a
# large, slow rebuild of the whole numpy/opencv/scipy stack; the standard,
# widely-documented workaround for exactly this collision is to disable the
# guard. Since this process never mixes threaded calls into both runtimes
# concurrently (embedding calls are the only OpenMP-parallel work), the
# known risk — silently wrong results from true concurrent use of both
# runtimes — does not apply here. Set only in this process, only where the
# conflict actually originates.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    hf_id = {
        "dinov2_vits14": "facebook/dinov2-small",
        "dinov2_vitb14": "facebook/dinov2-base",
    }.get(model_name, model_name)

    processor = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    model.eval()
    return processor, model, torch


def _pad_to_square(crop: np.ndarray, fill: int = 255) -> np.ndarray:
    """Pads with white to a square canvas, symbol centered, before the
    image reaches the HF processor.

    The DINOv2 processor resizes the shortest edge to 256 then centre-
    crops to 224x224. On an elongated crop — a resistor's tall/thin
    ink-tightened bbox, a capacitor's wide/short one — that step scales
    the short dimension up so much that the centre-crop discards most of
    the long dimension (a 20x83 crop's height would be reduced to ~21% of
    itself, likely losing both lead ends). Padding to square first means
    resize-then-crop sees the whole symbol instead of an arbitrary slice
    of it. Applies uniformly to both reference-library and candidate
    crops, since both paths share this one function.
    """
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = crop
    return canvas


_EMBEDDING_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768}


class Embedder:
    def __init__(self, model_name: str = "dinov2_vitb14"):
        self.model_name = model_name

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """crops: list of HxWx3 uint8 RGB arrays. Returns (N, dim) float32."""
        if not crops:
            dim = _EMBEDDING_DIMS.get(self.model_name, 768)
            return np.zeros((0, dim), dtype=np.float32)

        processor, model, torch = _load_model(self.model_name)
        squared = [_pad_to_square(c) for c in crops]
        inputs = processor(images=squared, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token pooled output as the crop-level embedding.
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)
