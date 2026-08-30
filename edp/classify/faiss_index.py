"""FAISS-backed nearest-neighbour index over the reference embedding
library.

At the library's current size (a few hundred augmented vectors) brute-
force `matrix @ query` is already sub-millisecond — FAISS doesn't buy
speed here. It buys the same interface at a size that *would* matter: the
whole design in docs/02/06/07 is "add a symbol class = drop in crops and
rebuild the index," and a linear scan stops being free once that library
grows past a few thousand entries (many classes x many rotation/mirror
variants x multiple exemplars per class). Building on FAISS from the
start means that scaling path costs nothing later — swapping IndexFlatIP
for an approximate index (IVF/HNSW) if the library ever gets that large
is a one-line change, not a rewrite.

IndexFlatIP (exact inner-product search) is used, not an approximate
index: embeddings are L2-normalised, so inner product is cosine
similarity, and at this scale exactness costs nothing.
"""
from __future__ import annotations

import numpy as np


class FaissLibraryIndex:
    def __init__(self, embeddings: np.ndarray):
        import faiss

        self.size = embeddings.shape[0]
        dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
        self._index = faiss.IndexFlatIP(dim) if dim else None
        if self._index is not None and self.size:
            self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def search(self, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices, scores), each shape (k,). Empty index ->
        empty arrays rather than raising, matching ReferenceLibrary's
        existing "empty library -> everything falls out as unknown"
        behaviour (see classify/match.py)."""
        if self._index is None or self.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        scores, indices = self._index.search(query, min(k, self.size))
        return indices[0], scores[0]
