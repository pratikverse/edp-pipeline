"""classify/library.py's embedding cache (_cache_signature,
_load_cached_embeddings, _save_cache) -- docs/08 section 1.3's "cost
found along the way" fix. Deliberately tests the cache helpers directly
rather than the full `ReferenceLibrary.build()`, which needs a real
(GPU-backed) DINOv2 model -- same "don't touch the embedder in unit
tests" precedent as test_classification_evidence.py."""
from pathlib import Path

import numpy as np

from edp.classify.library import _cache_signature, _load_cached_embeddings, _save_cache
from edp.config import ClassifyConfig


def _make_reference_dir(tmp_path: Path) -> Path:
    ref_dir = tmp_path / "reference"
    class_dir = ref_dir / "Resistor"
    class_dir.mkdir(parents=True)
    (class_dir / "R.png").write_bytes(b"not a real png, only mtime/size matter for the signature")
    return ref_dir


def test_signature_stable_when_nothing_changes(tmp_path):
    ref_dir = _make_reference_dir(tmp_path)
    cfg = ClassifyConfig()
    assert _cache_signature(ref_dir, cfg) == _cache_signature(ref_dir, cfg)


def test_signature_changes_when_a_file_is_touched(tmp_path):
    ref_dir = _make_reference_dir(tmp_path)
    cfg = ClassifyConfig()
    before = _cache_signature(ref_dir, cfg)

    img_path = ref_dir / "Resistor" / "R.png"
    img_path.write_bytes(b"different content, different size")

    assert _cache_signature(ref_dir, cfg) != before


def test_signature_changes_when_rotations_config_changes(tmp_path):
    ref_dir = _make_reference_dir(tmp_path)
    cfg_a = ClassifyConfig(rotations=[0, 90, 180, 270])
    cfg_b = ClassifyConfig(rotations=[0, 90])
    assert _cache_signature(ref_dir, cfg_a) != _cache_signature(ref_dir, cfg_b)


def test_cache_round_trip(tmp_path):
    ref_dir = _make_reference_dir(tmp_path)
    cfg = ClassifyConfig()
    signature = _cache_signature(ref_dir, cfg)
    meta = [("Resistor", 0, False, str(ref_dir / "Resistor" / "R.png"))]
    embeddings = np.random.default_rng(0).random((1, 8)).astype(np.float32)

    cache_path = tmp_path / "index.npz"
    _save_cache(cache_path, embeddings, meta, signature)

    loaded = _load_cached_embeddings(cache_path, signature, meta)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, embeddings)


def test_cache_miss_on_signature_mismatch(tmp_path):
    ref_dir = _make_reference_dir(tmp_path)
    cfg = ClassifyConfig()
    meta = [("Resistor", 0, False, str(ref_dir / "Resistor" / "R.png"))]
    embeddings = np.zeros((1, 8), dtype=np.float32)
    cache_path = tmp_path / "index.npz"
    _save_cache(cache_path, embeddings, meta, _cache_signature(ref_dir, cfg))

    assert _load_cached_embeddings(cache_path, "a-different-signature", meta) is None


def test_cache_miss_on_meta_mismatch(tmp_path):
    ref_dir = _make_reference_dir(tmp_path)
    cfg = ClassifyConfig()
    signature = _cache_signature(ref_dir, cfg)
    meta = [("Resistor", 0, False, str(ref_dir / "Resistor" / "R.png"))]
    embeddings = np.zeros((1, 8), dtype=np.float32)
    cache_path = tmp_path / "index.npz"
    _save_cache(cache_path, embeddings, meta, signature)

    different_meta = [("Capacitor", 0, False, str(ref_dir / "Resistor" / "R.png"))]
    assert _load_cached_embeddings(cache_path, signature, different_meta) is None


def test_cache_miss_when_file_does_not_exist(tmp_path):
    assert _load_cached_embeddings(tmp_path / "does_not_exist.npz", "sig", []) is None
