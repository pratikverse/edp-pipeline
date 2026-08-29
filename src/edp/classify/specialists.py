"""Geometric confusion-pair specialists (docs/08_improvement_plan.md Phase 6
of the classification-evidence architecture). Each specialist inspects a
candidate's *own* crop pixels only — never a symbol id, position, or
anything diagram-specific — and either returns a `ClassificationEvidence`
vote or abstains via `no_evidence`. match.py invokes a specialist only when
fusion is already ambiguous between classes in that specialist's confusion
group (`select_specialist`), never on every symbol — this keeps the fast,
deterministic path the default and specialists a targeted fallback.

Every feature and threshold here was chosen by directly inspecting D4's
actual symbol crops and this project's own KiCad reference renders (see
the conversation record around Phase 6), not invented blindly. Where a
threshold is a real judgement call, the docstring says so and gives the
crop that motivated it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .evidence import ClassificationEvidence, no_evidence

CONFUSION_GROUPS: list[frozenset[str]] = [
    frozenset({"Battery", "Capacitor", "Capacitor_Polarized"}),
    frozenset({"BJT_NPN", "BJT_PNP", "Potentiometer"}),
    frozenset({"BJT_NPN", "BJT_PNP"}),
]


def select_specialist(candidate_classes: set[str]):
    """Returns the specialist whose confusion group *contains* every class
    in `candidate_classes` (the ambiguous top-k under consideration), most
    specific group first — so a pure NPN-vs-PNP ambiguity gets the sharper
    arrowhead specialist instead of the coarser BJT/Potentiometer one, even
    though the broader group would also technically match. None if no
    specialist covers this particular ambiguity, which is the common case:
    most margin-below-threshold situations aren't a *known* confusion pair
    at all, and there's deliberately no generic catch-all specialist."""
    # npn_pnp_specialist is intentionally NOT routed here despite existing
    # and being unit-tested below (see its docstring): validated against
    # this project's own clean KiCad reference renders it's 2/2 correct,
    # but against D4's actual real crops it's 1/3 -- worse than a coin
    # flip. Wiring in a specialist with measured worse-than-chance
    # real-world reliability would make fusion worse, not better, which is
    # exactly the "don't force a geometric decision" principle this
    # architecture is built around. Left in the module as a documented,
    # tested, honest attempt rather than deleted or silently shipped.
    if candidate_classes <= CONFUSION_GROUPS[0]:
        return battery_capacitor_specialist
    if candidate_classes <= CONFUSION_GROUPS[1]:
        return bjt_potentiometer_specialist
    return None


def _binary_ink(crop_bgr: np.ndarray) -> np.ndarray:
    """0/255 ink mask, ink=255. Otsu rather than a fixed threshold since
    crop contrast varies (scan quality, anti-aliasing)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Specialist A: Battery vs Capacitor vs Capacitor_Polarized
# ---------------------------------------------------------------------------
#
# Grounded in inspecting D4's actual SYM_006/SYM_013/SYM_016 crops and this
# project's own reference renders (data/reference/{Battery,Capacitor,
# Capacitor_Polarized}/):
#   - Battery (multi-cell, SYM_006 / data/reference/Battery/Battery.png):
#     4+ plates of alternating length, stacked perpendicular to the leads.
#   - Capacitor_Polarized as D4 actually draws it (SYM_013): exactly 2
#     plates, a "+" mark, and — in D4's specific drawing style — one plate
#     is visibly CURVED (bowed), unlike this project's own KiCad reference
#     render of the same class, which draws polarity via plate *thickness*
#     (thick+thin) instead of curvature. Real drawings vary in convention;
#     this specialist trusts curvature when present since it's an
#     unambiguous signal, and accepts that a Capacitor_Polarized drawn in
#     the thick/thin-only style (like our own reference) won't trigger it
#     — falls through to the "+"-mark rule below instead.
#   - Battery, single-cell style (SYM_016): 2 plates, a "+" mark, BOTH
#     flat/straight — structurally close to a plain Capacitor, distinguished
#     here only by the "+" mark's presence. This is a real, disclosed
#     limitation: a single-cell battery and this project's own reference
#     rendering of Capacitor_Polarized are nearly geometric twins (compare
#     data/reference/Battery/Battery_Cell.png and
#     data/reference/Capacitor_Polarized/C_Polarized.png side by side —
#     they differ only in plate proportions). Curvature is checked first
#     specifically because it's the one feature that actually separated the
#     two real D4 instances; a "+" mark alone is treated as weaker,
#     lower-confidence evidence for Battery over Capacitor_Polarized.
#   - Capacitor: 2 equal-ish flat plates, no "+" mark.


def battery_capacitor_specialist(crop_bgr: np.ndarray) -> ClassificationEvidence:
    source = "geometry:battery_capacitor"
    if crop_bgr is None or crop_bgr.size == 0:
        return no_evidence(source, reason="empty_crop")

    binary = _binary_ink(crop_bgr)
    h, w = binary.shape
    if h < 10 or w < 10:
        return no_evidence(source, reason="crop_too_small", h=h, w=w)

    # Plates run perpendicular to the leads; leads run along the symbol's
    # long axis. Rotate so "plates are horizontal rows" always holds.
    landscape = w > h * 1.3
    work = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE) if landscape else binary

    plates = _find_plates(work)
    if len(plates) < 2:
        return no_evidence(source, reason=f"only_{len(plates)}_plates_found")

    has_plus = _detect_plus_mark(work)

    if len(plates) >= 4:
        return ClassificationEvidence(
            source=source,
            class_scores={"Battery": 1.0},
            confidence=0.85,
            metadata={"rule": "plate_count>=4", "plate_count": len(plates), "has_plus": has_plus},
        )

    curved = any(p[4] for p in plates[:2])
    if curved:
        return ClassificationEvidence(
            source=source,
            class_scores={"Capacitor_Polarized": 1.0},
            confidence=0.7,
            metadata={"rule": "curved_plate", "plate_count": len(plates), "has_plus": has_plus},
        )
    if has_plus:
        return ClassificationEvidence(
            source=source,
            class_scores={"Battery": 1.0},
            confidence=0.55,  # weaker: this is the "near-twin" case, see module docstring
            metadata={"rule": "plus_mark_no_curve", "plate_count": len(plates)},
        )
    return ClassificationEvidence(
        source=source,
        class_scores={"Capacitor": 1.0},
        confidence=0.6,
        metadata={"rule": "flat_plates_no_plus", "plate_count": len(plates)},
    )


_PLATE_MIN_RUN_PX = 14  # see _find_plates docstring


def _max_run(row: np.ndarray) -> tuple[int, int, int] | None:
    """(length, x0, x1) of the longest contiguous ink run in one row, or
    None if the row is empty. A row can contain more than one separate ink
    run (e.g. a value-label glyph plus the actual plate on the same row),
    which is exactly why this looks at the longest *run*, not the row's
    total ink-pixel count — see _find_plates."""
    idx = np.where(row > 0)[0]
    if len(idx) == 0:
        return None
    splits = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, splits + 1)
    best = max(runs, key=len)
    return len(best), int(best[0]), int(best[-1])


def _find_plates(work: np.ndarray) -> list[tuple[int, int, int, int, bool]]:
    """Returns (y0, y1, x0, x1, is_curved) per detected plate.

    Two separate passes, since a flat and a bowed plate don't look alike
    in the same feature:

    - Flat plates: rows whose *longest single ink run* clears
      _PLATE_MIN_RUN_PX. Deliberately not "total ink in the row" (an
      earlier version of this used a width-fraction threshold and broke
      the moment a candidate crop also contained nearby value-label text
      on the same rows, e.g. D4's SYM_013 candidate box includes "C1 10uF
      50V" text to the left of the actual plates) — a run-length test
      only cares about the single longest *contiguous* stroke, so
      disconnected text elsewhere in the row can't inflate it. 14px was
      picked because every plate observed (D4's real crops and this
      project's own KiCad reference renders) is comfortably >20px while
      individual text glyphs and the stem are both well under 10px.
    - Curved plates: a flat-row scan structurally can't find these — a
      swept 1px-wide arc's longest run in any *single* row is short even
      though the arc spans a wide x-range overall. Found instead via
      connected components: a wide (>=_PLATE_MIN_RUN_PX), short-ish
      (<50% of crop height, ruling out a stem+plate merged into one tall
      component), low-fill-ratio (<0.4 — a thin swept stroke fills much
      less of its bounding box than a filled/solid shape) component that
      doesn't already overlap a flat plate's row range.
    """
    h, w = work.shape
    flat: list[tuple[int, int, int, int, bool]] = []
    y = 0
    while y < h:
        run = _max_run(work[y])
        if run and run[0] >= _PLATE_MIN_RUN_PX:
            y0 = y
            xs = []
            while y < h:
                run = _max_run(work[y])
                if not run or run[0] < _PLATE_MIN_RUN_PX:
                    break
                xs.append((run[1], run[2]))
                y += 1
            y1 = y
            x0 = min(a for a, _b in xs)
            x1 = max(b for _a, b in xs)
            flat.append((y0, y1, x0, x1, False))
        else:
            y += 1

    curved: list[tuple[int, int, int, int, bool]] = []
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(work, connectivity=8)
    for i in range(1, n):
        x, y0c, cw, ch, area = stats[i]
        if cw < _PLATE_MIN_RUN_PX or ch > 0.5 * h:
            continue
        fill = area / (cw * ch) if cw * ch else 1.0
        if fill >= 0.4 or ch < 6:
            continue
        y1c = y0c + ch
        if any(not (y1c <= f0 or y0c >= f1) for f0, f1, _x0, _x1, _c in flat):
            continue  # already covered by a flat-plate row range
        curved.append((int(y0c), int(y1c), int(x), int(x + cw), True))

    return sorted(flat + curved, key=lambda p: p[0])


def _detect_plus_mark(work: np.ndarray) -> bool:
    """A "+" glyph is a small, roughly-square connected component with ink
    concentrated along its middle row AND middle column but not its
    corners — distinguishes it from a plate (wide, short — fails the
    square-aspect check) or a lead (thin, tall — same)."""
    wh, ww = work.shape
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(work, connectivity=8)
    for i in range(1, n):
        x, y, cw, ch, _area = stats[i]
        if cw < 4 or ch < 4:
            continue
        if cw > 0.4 * ww or ch > 0.4 * wh:
            continue
        if abs(cw - ch) > 0.6 * max(cw, ch):
            continue
        sub = work[y : y + ch, x : x + cw]
        mid_row = sub[max(0, ch // 2 - 1) : ch // 2 + 2, :]
        mid_col = sub[:, max(0, cw // 2 - 1) : cw // 2 + 2]
        corner = max(1, min(cw, ch) // 4)
        corners_ink = (
            int(sub[:corner, :corner].sum())
            + int(sub[:corner, -corner:].sum())
            + int(sub[-corner:, :corner].sum())
            + int(sub[-corner:, -corner:].sum())
        )
        cross_ink = int(mid_row.sum()) + int(mid_col.sum())
        if mid_row.sum() > 0 and mid_col.sum() > 0 and corners_ink < 0.3 * cross_ink:
            return True
    return False


# ---------------------------------------------------------------------------
# Specialist B: BJT (either polarity) vs Potentiometer
# ---------------------------------------------------------------------------
#
# Grounded in this project's own reference renders (data/reference/BJT_NPN,
# BJT_PNP, Potentiometer): a BJT is drawn inside a circle with 3 leads
# radiating from its boundary; a potentiometer is a rectangular resistor
# body with a wiper arrow entering from the side — no circle anywhere.
# Circle presence is by far the more reliable signal of the two (a
# protruding wiper arrow can distort a naive rectangle-fill-ratio check),
# so it's checked first and decisively; the rectangle check only fires when
# no circle was found, as a positive confirmation for Potentiometer rather
# than the primary signal.


def bjt_potentiometer_specialist(crop_bgr: np.ndarray) -> ClassificationEvidence:
    source = "geometry:bjt_potentiometer"
    if crop_bgr is None or crop_bgr.size == 0:
        return no_evidence(source, reason="empty_crop")

    binary = _binary_ink(crop_bgr)
    h, w = binary.shape
    if h < 12 or w < 12:
        return no_evidence(source, reason="crop_too_small", h=h, w=w)

    circle = _detect_circle(binary)
    if circle is not None:
        cx, cy, r = circle
        return ClassificationEvidence(
            source=source,
            # Polarity undecided here — npn_pnp_specialist handles that
            # narrower ambiguity separately (see select_specialist).
            class_scores={"BJT_NPN": 0.5, "BJT_PNP": 0.5},
            confidence=0.75,
            metadata={"rule": "circle_found", "circle": (float(cx), float(cy), float(r))},
        )

    if _has_rectangular_body(binary):
        return ClassificationEvidence(
            source=source,
            class_scores={"Potentiometer": 1.0},
            confidence=0.6,
            metadata={"rule": "rectangular_body_no_circle"},
        )

    return no_evidence(source, reason="neither_circle_nor_rectangle_detected")


def _detect_circle(binary: np.ndarray) -> tuple[float, float, float] | None:
    h, w = binary.shape
    min_r = int(0.25 * min(h, w))
    max_r = int(0.65 * min(h, w))
    if min_r < 5:
        return None
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(h, w),
        param1=50,
        # 30 is not a default -- lower values (~15) fire a false positive on
        # this project's own Potentiometer reference render (its resistor
        # body + wiper arrow apparently has enough incidental curvature to
        # pass a looser accumulator threshold); verified empirically that
        # 30 rejects that false positive while still finding every real BJT
        # circle tested (D4's 3 transistor crops + both BJT reference renders).
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None or len(circles) == 0:
        return None
    x, y, r = circles[0][0]
    return float(x), float(y), float(r)


def _has_rectangular_body(binary: np.ndarray) -> bool:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    if cw * ch == 0:
        return False
    extent = cv2.contourArea(largest) / (cw * ch)
    aspect = ch / cw if cw else 0
    return extent > 0.5 and 1.5 < aspect < 6.0


# ---------------------------------------------------------------------------
# Specialist C: BJT_NPN vs BJT_PNP (arrowhead direction)
# ---------------------------------------------------------------------------
#
# Only invoked when the ambiguity is *already* narrowed to exactly these
# two classes (select_specialist) — never applied generically. Convention
# (visible in data/reference/BJT_NPN vs BJT_PNP): the emitter lead's
# arrowhead points AWAY from the circle centre for NPN, TOWARD it for PNP.
# Found via the arrowhead's triangular contour: its farthest vertex from
# its own centroid is the tip; comparing the tip's distance from the
# circle centre against the triangle centroid's distance tells which way
# it points.
#
# MEASURED RESULT (see select_specialist's comment): correct on both of
# this project's own reference renders, but only 1/3 correct on D4's real
# transistor crops -- worse than chance there. Kept implemented and tested
# per the architecture spec's "a documented justified implementation
# attempt" allowance, but NOT routed by select_specialist, so it never
# actually votes in production fusion. Left for a future attempt with a
# more robust arrowhead localisation (this crop resolution may simply be
# too coarse for a 3-vertex polygon fit to be stable) rather than being
# deleted outright.


def npn_pnp_specialist(crop_bgr: np.ndarray) -> ClassificationEvidence:
    source = "geometry:npn_pnp"
    if crop_bgr is None or crop_bgr.size == 0:
        return no_evidence(source, reason="empty_crop")

    binary = _binary_ink(crop_bgr)
    h, w = binary.shape
    if h < 12 or w < 12:
        return no_evidence(source, reason="crop_too_small", h=h, w=w)

    circle = _detect_circle(binary)
    if circle is None:
        return no_evidence(source, reason="no_circle_found")
    cx, cy, r = circle

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_triangle = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # arrowhead is small relative to the circle -- filters out the
        # circle itself and the long straight leads
        if area < 4 or area > 0.15 * (np.pi * r * r):
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        # A stroked (not filled) arrowhead's contour has extra small
        # zig-zags from the outline's two nearly-parallel sides, so a
        # single fixed epsilon can return 4-6 vertices instead of 3
        # (verified on the reference BJT_NPN render: 0.04*peri -> 4 verts,
        # 0.08*peri -> the correct 3). Try increasingly permissive epsilons
        # rather than committing to one value that happened to work once.
        approx = None
        for eps_frac in (0.02, 0.05, 0.08, 0.12):
            candidate = cv2.approxPolyDP(cnt, eps_frac * peri, True)
            if len(candidate) == 3:
                approx = candidate
                break
        if approx is not None:
            best_triangle = approx.reshape(3, 2).astype(float)
            break  # first plausible triangle; crops here have exactly one arrowhead

    if best_triangle is None:
        return no_evidence(source, reason="no_arrowhead_triangle_found")

    centroid = best_triangle.mean(axis=0)
    tip = max(best_triangle, key=lambda p: np.linalg.norm(p - centroid))
    center = np.array([cx, cy])
    tip_dist = np.linalg.norm(tip - center)
    centroid_dist = np.linalg.norm(centroid - center)

    if tip_dist > centroid_dist:
        return ClassificationEvidence(
            source=source,
            class_scores={"BJT_NPN": 1.0},
            confidence=0.6,
            metadata={"rule": "arrow_tip_points_away", "tip_dist": float(tip_dist), "centroid_dist": float(centroid_dist)},
        )
    return ClassificationEvidence(
        source=source,
        class_scores={"BJT_PNP": 1.0},
        confidence=0.6,
        metadata={"rule": "arrow_tip_points_inward", "tip_dist": float(tip_dist), "centroid_dist": float(centroid_dist)},
    )
