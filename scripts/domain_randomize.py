"""Domain-randomization transforms shared by the synthetic dataset
generators (generate_synthetic_dataset.py, generate_ladder_circuits.py)
and generate_probe_training_data.py (docs/08_improvement_plan.md Phase
1.3 / the classification-evidence architecture's Phase 7 linear probe).

Every training image built from data/reference/ so far has been a *clean*
KiCad render — uniform 1px stroke, no noise, no blur, no scan artifacts —
while real circuit drawings are hand-drawn, scanned, printed, or exported
at varying quality. This is a diagnosed root cause behind both DINOv2's
weak performance (self-supervised on natural photographs, never seen a
schematic at all, let alone a noisy one) and part of YOLO's KiCad-vs-real
gap (docs/08 section 2.4).

Every transform here is deliberately kept semantics-preserving — a
jittered resistor is still unambiguously a resistor — per the spec's own
warning against transformations that could alter what a symbol reads as:
no free-form elastic warping, no non-uniform aspect distortion (would
change relative plate/lead proportions that some specialists key off of —
see classify/specialists.py), no rotation past a few degrees (that's what
the *existing* 0/90/180/270 handling in the generators is already for,
unchanged, since those really do represent different valid orientations).
"""
from __future__ import annotations

import random

import cv2
import numpy as np


def randomize_symbol(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Per-symbol jitter, applied before compositing/saving: a small
    rotation (drafting/scanning isn't perfectly axis-aligned, not a real
    orientation change) and a stroke-width dilate/erode (different
    rendering tools and pen weights produce different line thickness for
    the same symbol)."""
    img = _jitter_rotation(img, rng)
    img = _jitter_stroke_width(img, rng)
    return img


def randomize_canvas(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    """Whole-image jitter, applied last: blur, sensor/scan noise,
    contrast/gamma, and a simulated JPEG re-encode — these are page-level
    scan/export artifacts, not something that varies symbol-to-symbol."""
    canvas = _jitter_blur(canvas, rng)
    canvas = _jitter_noise(canvas, rng)
    canvas = _jitter_contrast_gamma(canvas, rng)
    canvas = _jitter_jpeg(canvas, rng)
    return canvas


def _jitter_rotation(img: np.ndarray, rng: random.Random, max_deg: float = 4.0) -> np.ndarray:
    angle = rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _jitter_stroke_width(img: np.ndarray, rng: random.Random) -> np.ndarray:
    # Weighted toward "none" (2/4 of the choices) so most examples still
    # look like a normally-rendered symbol -- jitter is meant to widen the
    # training distribution's tails, not shift its centre.
    choice = rng.choice(["thin", "none", "none", "thick"])
    if choice == "none":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = ((gray < 250).astype(np.uint8)) * 255
    kernel = np.ones((2, 2), np.uint8)
    ink = cv2.dilate(ink, kernel, iterations=1) if choice == "thick" else cv2.erode(ink, kernel, iterations=1)
    out = np.full_like(img, 255)
    out[ink > 0] = (0, 0, 0)
    return out


def _jitter_blur(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return canvas
    sigma = rng.uniform(0.3, 0.8)
    return cv2.GaussianBlur(canvas, (3, 3), sigma)


def _jitter_noise(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return canvas
    sigma = rng.uniform(2, 8)
    # np.random rather than `rng` (stdlib Random has no vectorised normal
    # draw); seeded from `rng` so the whole pipeline stays reproducible
    # from one seed.
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, sigma, canvas.shape)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jitter_contrast_gamma(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-15, 15)  # brightness
    out = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)
    gamma = rng.uniform(0.85, 1.15)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(out, table)


def _jitter_jpeg(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.6:
        return canvas
    quality = rng.randint(40, 85)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else canvas
