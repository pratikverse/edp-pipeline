"""Stage 4 - OCR (Tesseract multi-pass) and text/symbol association."""
from __future__ import annotations


# ===========================================================================
# ocr.py
# ===========================================================================

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

# ===========================================================================
# associate.py
# ===========================================================================

import re

from edp.config import TextConfig
from edp.types import BBox, Symbol, TextToken

_ID_RE = re.compile(r"^([A-Za-z]{1,4}\d{1,3})$")
_VALUE_RE = re.compile(r"^\d+(\.\d+)?\s*[a-zA-Zµ]*$")


def nearby_token_text(bbox: BBox, tokens: list[TextToken], cfg: TextConfig, limit: int = 6) -> str:
    """The same "closest tokens within max_association_distance, joined"
    logic `associate_tokens` uses for `symbol.ocr_text_raw`, exposed for
    any bbox (not just an already-classified Symbol's). limit=6, not 3:
    OCR now runs multiple Tesseract configs per orientation and doesn't
    dedup them (see text/ocr.py's run_ocr docstring) since picking a
    single "winning" reading measurably lost real designators — so a
    handful of near-identical readings of the same physical label
    commonly occupy the nearest few slots, and a limit of 3 could crowd
    out a second, genuinely different nearby label (e.g. a value string
    next to a designator) entirely. Used by
    classify/text_prior.py to get OCR evidence *before* classification has
    produced a Symbol — pipeline.py calls this once per candidate ahead of
    the classify stage. Single source of truth for "which tokens count as
    near this box," so the pre-classification hint and the final
    `ocr_text_raw` never disagree about what's nearby."""
    nearby = [t for t in tokens if _distance(bbox, t.bbox) <= cfg.max_association_distance]
    nearby.sort(key=lambda t: _distance(bbox, t.bbox))
    return " ".join(t.text for t in nearby[:limit])


def associate_tokens(symbols: list[Symbol], tokens: list[TextToken], cfg: TextConfig) -> list[Symbol]:
    """Mutates and returns `symbols` with ocr_text_raw/value/id assigned
    from the nearest token within max_association_distance.

    Instance ids ("R1" vs "R2") are assigned by *global* greedy nearest-
    match, not independently per symbol: with density-based localization,
    several candidate bboxes can cluster around one physical label, and an
    independent per-symbol "closest token" lookup lets two different
    symbols both claim the same id ("R4" assigned twice), which corrupts
    every downstream `connections` reference. Greedy assignment consumes
    each id token at most once.
    """
    for symbol in symbols:
        text = nearby_token_text(symbol.bbox, tokens, cfg)
        if text:
            symbol.ocr_text_raw = text

    _assign_ids_greedily(symbols, tokens, cfg)
    _assign_values(symbols, tokens, cfg)
    return symbols


def _assign_ids_greedily(symbols: list[Symbol], tokens: list[TextToken], cfg: TextConfig) -> None:
    candidates = []  # (distance, symbol, id_text)
    for symbol in symbols:
        for token in tokens:
            dist = _distance(symbol.bbox, token.bbox)
            if dist > cfg.max_association_distance:
                continue
            for part in token.text.split():
                if _ID_RE.match(part):
                    candidates.append((dist, symbol, part))

    candidates.sort(key=lambda c: c[0])
    claimed_symbols: set[int] = set()
    claimed_ids: set[str] = set()
    for _dist, symbol, id_text in candidates:
        if id(symbol) in claimed_symbols or id_text in claimed_ids:
            continue
        symbol.id = id_text
        symbol.label_source = "ocr"
        for terminal in symbol.terminals:
            terminal.symbol_id = id_text
        claimed_symbols.add(id(symbol))
        claimed_ids.add(id_text)


def _assign_values(symbols: list[Symbol], tokens: list[TextToken], cfg: TextConfig) -> None:
    # Values are not required to be unique (several inductors can share
    # "50uH"), so this stays a simple per-symbol nearest-match.
    for symbol in symbols:
        nearby = [t for t in tokens if _distance(symbol.bbox, t.bbox) <= cfg.max_association_distance]
        nearby.sort(key=lambda t: _distance(symbol.bbox, t.bbox))
        for token in nearby:
            for part in token.text.split():
                if _VALUE_RE.match(part) and not _ID_RE.match(part) and symbol.value is None:
                    symbol.value = part


def _distance(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    # edge-to-edge gap, not centre-to-centre, so a large symbol doesn't
    # unfairly penalise a nearby label
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    if dx == 0 and dy == 0:
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    return (dx ** 2 + dy ** 2) ** 0.5
