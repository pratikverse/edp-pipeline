"""Real-data second localizer (edp/realdetect.py). Network paths are not
exercised — only the graceful-degradation and response-parsing logic."""
from __future__ import annotations

import numpy as np

from edp.realdetect import RealDetectorSpec, detect_real


def test_spec_from_dict_none():
    assert RealDetectorSpec.from_dict(None) is None
    assert RealDetectorSpec.from_dict({}) is None


def test_spec_from_dict():
    spec = RealDetectorSpec.from_dict(
        {"provider": "roboflow", "model_id": "a/b/2", "conf": 0.4}
    )
    assert spec.model_id == "a/b/2"
    assert spec.conf == 0.4
    assert spec.overlap == 0.30  # default


def test_detect_real_abstains_without_provider():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    spec = RealDetectorSpec(provider="unknown", model_id="x/y/1")
    assert detect_real(img, spec) == []


def test_detect_real_parses_cached_response(tmp_path, monkeypatch):
    """A cached Roboflow-shape response is turned into class-agnostic
    Candidates in image coordinates."""
    import edp.realdetect as rd

    monkeypatch.setattr(rd, "_CACHE_DIR", tmp_path)
    spec = RealDetectorSpec(provider="roboflow", model_id="a/b/2", conf=0.3, overlap=0.3)
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # Pre-seed the cache so no network call happens.
    import hashlib
    import json

    key = hashlib.sha1(
        img.tobytes() + spec.model_id.encode() + f"{spec.conf}:{spec.overlap}".encode()
    ).hexdigest()[:16]
    (tmp_path / f"{key}.json").write_text(
        json.dumps(
            {"predictions": [
                {"x": 100, "y": 50, "width": 40, "height": 20, "confidence": 0.9, "class": "resistor"},
                {"x": 0, "y": 0, "width": 4, "height": 4, "confidence": 0.5, "class": "diode"},
            ]}
        )
    )

    cands = detect_real(img, spec)
    assert len(cands) == 2
    c = cands[0]
    assert c.source == "real_detector"
    assert c.yolo_class is None  # class-agnostic
    assert c.yolo_confidence == 0.9
    assert c.bbox == (80, 40, 120, 60)
