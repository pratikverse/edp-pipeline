"""Real-data second localizer — an off-the-shelf circuit detector trained
on real hand-drawn / photographed circuits, run *class-agnostically* as a
recall booster alongside the in-domain synthetic YOLO.

Rationale (docs/09 §2.1 territory): the synthetic detector is only as good
as its synthetic training data and is the weakest link on unfamiliar
real-world symbol styles. A detector trained on real circuits sees drawing
noise, line-weight variation and conventions the synthetic one never did.
We take only its *boxes*, never its class labels — its taxonomy won't match
ours and a mislabel would actively hurt fusion. Its proposals are merged
into the same candidate list (edp/localize.merge_overlapping); a box it
alone found still gets classified normally by DINOv2 + probe + OCR, just
with no YOLO class vote.

Provider: Roboflow hosted inference. This is a network call, so:
  - responses are cached to disk keyed by image content — a second run on
    the same drawing is offline and deterministic;
  - any failure (no API key, offline, HTTP error) logs once and returns
    [], exactly like a missing model file elsewhere in the pipeline. The
    pipeline never depends on it being reachable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from edp.types import BBox, Candidate

log = logging.getLogger(__name__)

_CACHE_DIR = Path("outputs/realdetect_cache")
_warned = False


@dataclass(frozen=True)
class RealDetectorSpec:
    """The `real_detector:` block of a domain pack's pack.yaml."""

    provider: str            # "roboflow" (hosted) or "local" (an ultralytics .pt)
    model_id: str            # roboflow "ws/project/version", or a local weights path
    conf: float = 0.20       # 0-1; roboflow wants it *100, ultralytics wants it as-is
    overlap: float = 0.30

    @classmethod
    def from_dict(cls, d: dict | None) -> "RealDetectorSpec | None":
        if not d:
            return None
        return cls(
            provider=d.get("provider", "roboflow"),
            model_id=d.get("model_id") or d["weights"],
            conf=float(d.get("conf", 0.20)),
            overlap=float(d.get("overlap", 0.30)),
        )


def detect_real(img_rgb: np.ndarray, spec: RealDetectorSpec) -> list[Candidate]:
    """Class-agnostic candidate boxes from the off-the-shelf detector, or
    [] on any failure. `img_rgb` is the same array the synthetic detector
    gets, so the two proposal sets share a coordinate frame."""
    global _warned
    if spec is None:
        return []
    if spec.provider == "local":
        return _detect_local(img_rgb, spec)
    if spec.provider != "roboflow":
        return []

    raw = _predict_cached(img_rgb, spec)
    if raw is None:
        if not _warned:
            log.warning("real_detector (%s) unavailable — continuing without it", spec.model_id)
            _warned = True
        return []

    h, w = img_rgb.shape[:2]
    out: list[Candidate] = []
    for p in raw.get("predictions", []):
        cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        bbox: BBox = (x0, y0, x1, y1)
        # Deliberately class-agnostic: no yolo_class. yolo_confidence carries
        # the detector's own score only so merge_overlapping has a tiebreak
        # value; a class-bearing synthetic box still wins the class slot.
        out.append(Candidate(bbox=bbox, kind="symbol", yolo_confidence=float(p.get("confidence", 0.0)),
                             source="real_detector"))
    return out


_local_model_cache: dict[str, object] = {}


def _detect_local(img_rgb: np.ndarray, spec: RealDetectorSpec) -> list[Candidate]:
    """Class-agnostic boxes from a local ultralytics .pt (e.g. the
    single-class detector from scripts/build_realdata_detector.py)."""
    global _warned
    weights = Path(spec.model_id)
    if not weights.exists():
        if not _warned:
            log.warning("real_detector local weights missing: %s — continuing without it", weights)
            _warned = True
        return []
    try:
        from ultralytics import YOLO

        model = _local_model_cache.get(str(weights))
        if model is None:
            model = YOLO(str(weights))
            _local_model_cache[str(weights)] = model
        res = model.predict(img_rgb, conf=spec.conf, iou=spec.overlap, verbose=False)[0]
    except Exception as e:
        log.warning("local real_detector failed: %s", e)
        return []

    h, w = img_rgb.shape[:2]
    out: list[Candidate] = []
    for box in res.boxes:
        x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(Candidate(bbox=(x0, y0, x1, y1), kind="symbol",
                             yolo_confidence=float(box.conf[0]), source="real_detector"))
    return out


def _predict_cached(img_rgb: np.ndarray, spec: RealDetectorSpec) -> dict | None:
    key = hashlib.sha1(
        img_rgb.tobytes() + spec.model_id.encode() + f"{spec.conf}:{spec.overlap}".encode()
    ).hexdigest()[:16]
    cache_file = _CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    result = _predict_roboflow(img_rgb, spec)
    if result is not None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result))
    return result


def _predict_roboflow(img_rgb: np.ndarray, spec: RealDetectorSpec) -> dict | None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        _load_env_file()
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        return None
    try:
        import tempfile

        from roboflow import Roboflow

        ws, proj, ver = spec.model_id.split("/")
        rf = Roboflow(api_key=api_key)
        model = rf.workspace(ws).project(proj).version(int(ver)).model

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name
        try:
            return model.predict(
                tmp_path, confidence=int(spec.conf * 100), overlap=int(spec.overlap * 100)
            ).json()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:  # network, auth, quota, API shape change
        log.warning("roboflow predict failed: %s", e)
        return None


def _load_env_file() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
