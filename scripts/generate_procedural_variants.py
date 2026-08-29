"""Procedurally-drawn structural variants for classes where the diversity
that matters isn't rendering style (that's data/kicad_raw/ +
build_reference_from_kicad.py's job) but a *count* or *shape parameter* no
single fixed KiCad symbol can give us: how many battery cells, how many
resistor zigzag peaks, whether a polarized capacitor's second plate is
drawn flat or bowed. docs/08_improvement_plan.md section 1.3 / the
classification-evidence architecture's "wider synthetic base" work.

Writes directly to data/reference/<class>/procedural_*.png with a matching
*.terminals.json sidecar, in exactly the format
classify/library.py's `_load_terminals` already expects — no new loading
code needed, `ReferenceLibrary.build()` picks these up automatically
alongside the KiCad-sourced ones.

Every shape stays a recognisable, real-world instance of its class (same
"semantics-preserving" rule as scripts/domain_randomize.py) — this isn't
inventing new symbol conventions, it's covering the range of an existing
one that a single fixed rendering can't (a real battery might have 2 cells
or 6; a real resistor's zigzag might have 3 peaks or 5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"

BLACK = (0, 0, 0)
LINE = 2


def _new_canvas(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _save(class_name: str, name: str, img: np.ndarray, terminals: list[dict]) -> None:
    out_dir = REFERENCE_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{name}.png"), img)
    (out_dir / f"{name}.terminals.json").write_text(json.dumps(terminals, indent=2), encoding="utf-8")
    print(f"[procedural] {class_name}/{name}.png  ({len(terminals)} terminals)")


def draw_battery(cell_count: int) -> tuple[np.ndarray, list[dict]]:
    """Alternating long/short plates, `cell_count` of them, stacked
    vertically -- the real KiCad convention (data/reference/Battery/
    Battery.png) generalised to any cell count instead of the one fixed
    example we had. Odd cell counts are valid and common (e.g. a 3-cell
    9V-ish stack), not just the even counts KiCad ships fixed renders for."""
    plate_gap = 14
    lead_len = 20
    long_w, short_w = 60, 30
    w = 90
    h = 2 * lead_len + cell_count * plate_gap + 10
    img = _new_canvas(w, h)
    cx = w // 2

    y = lead_len
    cv2.line(img, (cx, 0), (cx, y), BLACK, LINE)
    for i in range(cell_count):
        plate_w = long_w if i % 2 == 0 else short_w
        cv2.line(img, (cx - plate_w // 2, y), (cx + plate_w // 2, y), BLACK, LINE)
        y += plate_gap
    y -= plate_gap
    cv2.line(img, (cx, y), (cx, y + lead_len), BLACK, LINE)

    # "+" mark near the first (long, positive) plate, offset to the side --
    # same convention as every real battery reference crop inspected this
    # session (see classify/specialists.py's battery_capacitor_specialist
    # docstring).
    px, py = cx + long_w // 2 + 12, lead_len - 2
    cv2.line(img, (px - 5, py), (px + 5, py), BLACK, 1)
    cv2.line(img, (px, py - 5), (px, py + 5), BLACK, 1)

    terminals = [
        {"name": "1", "point": [cx, 0], "body": [cx, lead_len]},
        {"name": "2", "point": [cx, h - 1], "body": [cx, y]},
    ]
    return img, terminals


def draw_zigzag_resistor(peak_count: int) -> tuple[np.ndarray, list[dict]]:
    """ANSI zigzag with `peak_count` peaks (data/reference/Resistor/R_US.png
    fixes this at a specific count) -- real resistor symbols vary this
    with drawing tool/scale, commonly 3-5 peaks."""
    lead_len = 25
    zigzag_h = peak_count * 14
    half_width = 12
    w = 2 * half_width + 10
    h = 2 * lead_len + zigzag_h
    img = _new_canvas(w, h)
    cx = w // 2

    y0 = lead_len
    cv2.line(img, (cx, 0), (cx, y0), BLACK, LINE)

    step = zigzag_h / (2 * peak_count)
    pts = [(cx, y0)]
    y = y0
    for i in range(2 * peak_count):
        y += step
        x = cx + (half_width if i % 2 == 0 else -half_width)
        pts.append((int(x), int(y)))
    pts.append((cx, y0 + zigzag_h))
    for p1, p2 in zip(pts, pts[1:]):
        cv2.line(img, p1, p2, BLACK, LINE)

    y_end = y0 + zigzag_h
    cv2.line(img, (cx, y_end), (cx, y_end + lead_len), BLACK, LINE)

    terminals = [
        {"name": "1", "point": [cx, 0], "body": [cx, y0]},
        {"name": "2", "point": [cx, h - 1], "body": [cx, y_end]},
    ]
    return img, terminals


def draw_curved_polarized_capacitor() -> tuple[np.ndarray, list[dict]]:
    """Flat top plate ("+") + bowed bottom plate -- the convention D4
    actually draws (see classify/specialists.py's battery_capacitor_
    specialist docstring), distinct from this project's existing KiCad
    reference (data/reference/Capacitor_Polarized/C_Polarized.png), which
    encodes polarity via plate *thickness* instead. Both conventions are
    real; the library should recognise either."""
    w, h = 90, 130
    img = _new_canvas(w, h)
    cx = w // 2
    lead_len = 30
    plate_w = 50

    y0 = lead_len
    cv2.line(img, (cx, 0), (cx, y0), BLACK, LINE)
    cv2.line(img, (cx - plate_w // 2, y0), (cx + plate_w // 2, y0), BLACK, LINE)

    px, py = cx + plate_w // 2 + 12, y0 - 5
    cv2.line(img, (px - 5, py), (px + 5, py), BLACK, 1)
    cv2.line(img, (px, py - 5), (px, py + 5), BLACK, 1)

    arc_y = y0 + 20
    cv2.ellipse(img, (cx, arc_y - 18), (plate_w // 2, 22), 0, 20, 160, BLACK, LINE)
    cv2.line(img, (cx, arc_y), (cx, h - 1), BLACK, LINE)

    terminals = [
        {"name": "1", "point": [cx, 0], "body": [cx, y0]},
        {"name": "2", "point": [cx, h - 1], "body": [cx, arc_y]},
    ]
    return img, terminals


def main() -> None:
    for n in (2, 3, 4, 5, 6):
        img, terminals = draw_battery(n)
        _save("Battery", f"procedural_{n}cell", img, terminals)

    for n in (3, 4, 5, 6):
        img, terminals = draw_zigzag_resistor(n)
        _save("Resistor", f"procedural_{n}peak", img, terminals)

    img, terminals = draw_curved_polarized_capacitor()
    _save("Capacitor_Polarized", "procedural_curved", img, terminals)


if __name__ == "__main__":
    sys.exit(main())
