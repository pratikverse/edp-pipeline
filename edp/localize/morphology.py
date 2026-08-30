"""Morphological ops that separate "symbol" regions from "wire" regions.
See docs/01_architecture_overview.md stage 2.

A `close_dashed_boundaries` helper (dilate-then-erode to bridge a dashed
rectangle's gaps, e.g. D5's SHIELD box) was planned alongside this but
never wired into the localization path that shipped — removed rather than
kept as unreferenced dead code; `dash_close_kernel` is gone from
LocalizeConfig for the same reason. Dashed-boundary handling remains a
real, open limitation (docs/01), just not addressed by this function."""
from __future__ import annotations

import cv2
import numpy as np

from edp.config import LocalizeConfig


def thick_symbol_mask(binary: np.ndarray, cfg: LocalizeConfig) -> np.ndarray:
    """Isolates thick/enclosed regions (candidate symbol bodies) via
    morphological opening: a kernel wider than typical wire thickness
    erases thin strokes entirely while thick shapes survive. This is what
    breaks the connectivity between a symbol and the wires touching it, so
    each symbol becomes its own connected component — the whole point of
    the wire/symbol split (docs/01 stage 2). Contours must be extracted
    from *this* mask, not from the full binary, or a drawing where every
    symbol is wired to its neighbours collapses into one giant blob."""
    k = cfg.thin_stroke_max_width
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.erode(binary, kernel, iterations=1)
    opened = cv2.dilate(opened, kernel, iterations=1)
    return opened


def thin_stroke_mask(binary: np.ndarray, cfg: LocalizeConfig) -> np.ndarray:
    """Wire-only mask: pixels present in `binary` but not in the thick
    symbol mask were part of a stroke thinner than the kernel."""
    thick = thick_symbol_mask(binary, cfg)
    return cv2.subtract(binary, thick)
