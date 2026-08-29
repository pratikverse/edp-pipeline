"""OCR pass: upscale then multi-orientation Tesseract. See docs/01 stage 4
and docs/02 (why upscaling is the dominant lever at this resolution)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from edp.config import TextConfig
from edp.types import TextToken


def _ensure_tessdata_prefix() -> None:
    """The conda-forge `tesseract` package (this project's local install)
    doesn't register TESSDATA_PREFIX itself, and without it Tesseract
    fails closed -- not with an exception (pytesseract's own try/except
    below swallows it), but by silently returning zero tokens, which
    quietly degrades every OCR-dependent stage (id assignment, and the
    text_prior classification evidence) without any visible error. Same
    "set the env var defensively rather than requiring activation" pattern
    already used for the MKL/OpenMP conflict in classify/embedder.py.
    Only sets it if unset and a real tessdata dir is found — never
    overrides an explicit environment choice."""
    if os.environ.get("TESSDATA_PREFIX"):
        return
    for candidate in (
        Path(sys.prefix) / "share" / "tessdata",
        Path(sys.prefix) / "Library" / "share" / "tessdata",
    ):
        if (candidate / "eng.traineddata").exists():
            os.environ["TESSDATA_PREFIX"] = str(candidate)
            return

_ROT_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
}


def run_ocr(gray: np.ndarray, cfg: TextConfig) -> list[TextToken]:
    """Returns text tokens in the *original* (un-upscaled, un-rotated)
    image's coordinate space.

    If Tesseract is not installed, returns an empty list rather than
    raising — the pipeline stays runnable, and downstream symbols simply
    fall back to auto-generated ids (`metadata.source` marks this).
    """
    try:
        import pytesseract
    except ImportError:
        return []

    _ensure_tessdata_prefix()

    h, w = gray.shape[:2]
    upscaled = cv2.resize(
        gray, (w * cfg.upscale_factor, h * cfg.upscale_factor), interpolation=cv2.INTER_LANCZOS4
    )

    # No dedup across the multiple configs/orientations, deliberately: a
    # first attempt greedily kept only the highest-confidence reading per
    # overlapping bbox, and it *lost* accuracy on D4 (the correct "$1"->"S1"
    # reading for a switch's designator was crowded out by a higher-
    # confidence but wrong reading of the same region from a different PSM
    # pass) -- Tesseract's confidence score for short, garbled, technical
    # tokens isn't a reliable "which reading is right" signal. Passing every
    # reading through and letting text/associate.py's nearest-token join
    # naturally surface duplicates of the correct reading (which then simply
    # confirms itself, redundantly but harmlessly) measured better than
    # picking a single "winner" per region.
    tokens: list[TextToken] = []
    for orientation in cfg.orientations:
        rotated = _rotate(upscaled, orientation)
        for tesseract_config in cfg.tesseract_configs:
            try:
                data = pytesseract.image_to_data(
                    rotated, config=tesseract_config, output_type=pytesseract.Output.DICT
                )
            except Exception:
                continue
            tokens.extend(_extract_tokens(data, rotated.shape, orientation, upscaled.shape, cfg.upscale_factor))
    return tokens


def _rotate(img: np.ndarray, orientation: int) -> np.ndarray:
    code = _ROT_CODE.get(orientation)
    return img if code is None else cv2.rotate(img, code)


def _extract_tokens(
    data: dict, rotated_shape, orientation: int, upscaled_shape, upscale_factor: int
) -> list[TextToken]:
    tokens: list[TextToken] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i].strip()
        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox_rot = (x, y, x + w, y + h)
        bbox_up = _unrotate_bbox(bbox_rot, rotated_shape, orientation, upscaled_shape)
        bbox_orig = tuple(v // upscale_factor for v in bbox_up)
        tokens.append(TextToken(text=text, bbox=bbox_orig, confidence=conf / 100.0, orientation=orientation))
    return tokens


def _unrotate_bbox(bbox, rotated_shape, orientation: int, target_shape):
    """Maps a bbox in the rotated image back to the upscaled (0deg) frame."""
    x0, y0, x1, y1 = bbox
    rh, rw = rotated_shape[:2]
    if orientation == 0:
        return (x0, y0, x1, y1)
    if orientation == 90:
        # rotated = original rotated 90 CW; invert: (x,y) -> (y, rh-1-x) style
        return (y0, rw - x1, y1, rw - x0)
    if orientation == 270:
        return (rh - y1, x0, rh - y0, x1)
    if orientation == 180:
        return (rw - x1, rh - y1, rw - x0, rh - y0)
    return (x0, y0, x1, y1)
