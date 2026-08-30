"""Linear-probe evidence source (docs/08_improvement_plan.md Phase 7): a
lightweight classifier (scripts/train_linear_probe.py) trained on frozen
DINOv2 embeddings of domain-randomized synthetic crops, as an
independently-evaluable alternative to nearest-neighbour matching
(classify/library.py). Reuses the exact same embedding already computed
for the `dinov2` evidence source in match.py — no extra embedding cost,
and the probe never runs at all if its artifact hasn't been trained
(`no_evidence`, same "stay runnable with an incomplete model" pattern as
the empty-reference-library case in classify/library.py).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .evidence import ClassificationEvidence, no_evidence

SOURCE = "dinov2_probe"


@lru_cache(maxsize=1)
def _load(model_path: str):
    import joblib

    if not model_path:
        return None
    path = Path(model_path)
    if not path.is_file():
        return None
    return joblib.load(path)


def probe_evidence(embedding: np.ndarray, model_path: str, top_k: int = 3) -> ClassificationEvidence:
    clf = _load(model_path)
    if clf is None:
        return no_evidence(SOURCE, reason="no_probe_model", model_path=model_path)

    proba = clf.predict_proba(embedding.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1][:top_k]
    class_scores = {str(clf.classes_[i]): float(proba[i]) for i in order}
    return ClassificationEvidence(
        source=SOURCE,
        class_scores=class_scores,
        # The probe's predicted probabilities already are calibrated-ish
        # per-class scores (softmax output) -- confidence=1.0 trusts them
        # directly, same treatment as the `dinov2` cosine-similarity scores
        # in match.py, rather than double-discounting by an extra scalar.
        confidence=1.0,
        metadata={"topk": [(str(clf.classes_[i]), round(float(proba[i]), 3)) for i in order]},
    )
