"""Procedural ISA-5.1 P&ID symbol reference library.

P&ID symbols are geometrically simple (circles, rectangles, triangles,
zigzags), so — exactly as scripts/generate_procedural_variants.py does for
electronic style variants — they are drawn directly rather than sourced
from an SVG library and rasterised. This gives full control over the
terminal (process-connection) points, which for process equipment are far
less standardised than KiCad's pin geometry.

Classes are the equipment actually present in D1-D3 (separator/P&ID
drawings): vessels, centrifugal pumps, gate/control/check valves,
instrument bubbles, heat exchangers, compressors/blowers, filters, motors.

    python scripts/build_pid_reference.py

Output: data/pid_reference/<Class>/<name>.png  + <name>.terminals.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "pid_reference"
S = 120  # canvas side
INK = 0
BG = 255
LW = 2


def _canvas() -> np.ndarray:
    return np.full((S, S, 3), BG, np.uint8)


def _poly(img, pts, fill=False):
    a = np.array(pts, np.int32).reshape(-1, 1, 2)
    if fill:
        cv2.fillPoly(img, [a], (INK, INK, INK))
    else:
        cv2.polylines(img, [a], True, (INK, INK, INK), LW)


def _line(img, p1, p2):
    cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), (INK, INK, INK), LW)


def _circle(img, c, r, filled=False):
    cv2.circle(img, tuple(map(int, c)), int(r), (INK, INK, INK), -1 if filled else LW)


def _zigzag(img, x0, x1, y, amp, n):
    step = (x1 - x0) / (2 * n)
    pts = [(x0, y)]
    for i in range(1, 2 * n):
        pts.append((x0 + i * step, y - amp if i % 2 else y + amp))
    pts.append((x1, y))
    cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (INK, INK, INK), LW)


# each builder returns (image, terminals) where a terminal is
# {"name","point":[x,y] wire-contact, "body":[x,y] inward anchor}
def gate_valve():
    img = _canvas()
    cy = S // 2
    _poly(img, [(15, cy - 28), (15, cy + 28), (S // 2, cy)])
    _poly(img, [(S - 15, cy - 28), (S - 15, cy + 28), (S // 2, cy)])
    t = [
        {"name": "A", "point": [15, cy], "body": [S // 2, cy]},
        {"name": "B", "point": [S - 15, cy], "body": [S // 2, cy]},
    ]
    return img, t


def control_valve():
    img, t = gate_valve()
    cx, cy = S // 2, S // 2
    _line(img, (cx, cy - 28), (cx, 26))          # stem
    cv2.ellipse(img, (cx, 22), (20, 12), 0, 180, 360, (INK, INK, INK), LW)  # diaphragm actuator
    return img, t


def check_valve():
    img, t = gate_valve()
    cx, cy = S // 2, S // 2
    _line(img, (cx + 12, cy - 18), (cx - 6, cy + 18))  # flapper
    _circle(img, (cx + 12, cy - 18), 4, filled=True)
    return img, t


def centrifugal_pump():
    img = _canvas()
    cx, cy, r = 52, 66, 34
    _circle(img, (cx, cy), r)
    # tangential discharge box up to top
    _poly(img, [(cx - 12, cy - r + 4), (cx - 12, 14), (cx + 14, 14), (cx + 14, cy - 20)])
    t = [
        {"name": "suction", "point": [cx - r, cy], "body": [cx, cy]},
        {"name": "discharge", "point": [cx + 1, 14], "body": [cx + 1, 40]},
    ]
    return img, t


def vessel_vertical():
    img = _canvas()
    x0, x1, y0, y1 = 34, 86, 20, 100
    _line(img, (x0, y0), (x0, y1))
    _line(img, (x1, y0), (x1, y1))
    cv2.ellipse(img, ((x0 + x1) // 2, y0), ((x1 - x0) // 2, 12), 0, 180, 360, (INK, INK, INK), LW)
    cv2.ellipse(img, ((x0 + x1) // 2, y1), ((x1 - x0) // 2, 12), 0, 0, 180, (INK, INK, INK), LW)
    cx = (x0 + x1) // 2
    t = [
        {"name": "top", "point": [cx, y0 - 12], "body": [cx, y0 + 10]},
        {"name": "bottom", "point": [cx, y1 + 12], "body": [cx, y1 - 10]},
        {"name": "side", "point": [x1, (y0 + y1) // 2], "body": [x0, (y0 + y1) // 2]},
    ]
    return img, t


def vessel_horizontal():
    img = _canvas()
    x0, x1, y0, y1 = 18, 102, 38, 82
    _line(img, (x0, y0), (x1, y0))
    _line(img, (x0, y1), (x1, y1))
    cv2.ellipse(img, (x0, (y0 + y1) // 2), (12, (y1 - y0) // 2), 0, 90, 270, (INK, INK, INK), LW)
    cv2.ellipse(img, (x1, (y0 + y1) // 2), (12, (y1 - y0) // 2), 0, -90, 90, (INK, INK, INK), LW)
    cy = (y0 + y1) // 2
    t = [
        {"name": "left", "point": [x0 - 12, cy], "body": [x0 + 10, cy]},
        {"name": "right", "point": [x1 + 12, cy], "body": [x1 - 10, cy]},
        {"name": "top", "point": [(x0 + x1) // 2, y0], "body": [(x0 + x1) // 2, cy]},
        {"name": "bottom", "point": [(x0 + x1) // 2, y1], "body": [(x0 + x1) // 2, cy]},
    ]
    return img, t


def heat_exchanger():
    img = _canvas()
    cx, cy, r = S // 2, S // 2, 42
    _circle(img, (cx, cy), r)
    _zigzag(img, cx - r + 4, cx + r - 4, cy, 12, 3)
    t = [
        {"name": "A", "point": [cx - r, cy], "body": [cx, cy]},
        {"name": "B", "point": [cx + r, cy], "body": [cx, cy]},
        {"name": "C", "point": [cx, cy - r], "body": [cx, cy]},
        {"name": "D", "point": [cx, cy + r], "body": [cx, cy]},
    ]
    return img, t


def shell_tube_exchanger():
    img = _canvas()
    x0, x1, y0, y1 = 16, 104, 40, 80
    _poly(img, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    for yy in (y0 + 10, (y0 + y1) // 2, y1 - 10):
        _line(img, (x0, yy), (x1, yy))
    cy = (y0 + y1) // 2
    t = [
        {"name": "shell_in", "point": [x0, y0 + 6], "body": [(x0 + x1) // 2, y0 + 6]},
        {"name": "shell_out", "point": [x1, y1 - 6], "body": [(x0 + x1) // 2, y1 - 6]},
        {"name": "tube_in", "point": [x0, cy], "body": [(x0 + x1) // 2, cy]},
        {"name": "tube_out", "point": [x1, cy], "body": [(x0 + x1) // 2, cy]},
    ]
    return img, t


def compressor():
    img = _canvas()
    x0, x1 = 22, 98
    _poly(img, [(x0, 24), (x1, 12), (x1, S - 12), (x0, S - 24)])
    cy = S // 2
    t = [
        {"name": "in", "point": [x0, cy], "body": [x1, cy]},
        {"name": "out", "point": [x1, cy], "body": [x0, cy]},
    ]
    return img, t


def filter_sym():
    img = _canvas()
    x0, x1, y0, y1 = 28, 92, 28, 92
    _poly(img, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    for i in range(-2, 3):
        _line(img, (x0, (y0 + y1) // 2 + i * 14), (x1, (y0 + y1) // 2 + i * 14 - 20))
    cy = (y0 + y1) // 2
    t = [
        {"name": "in", "point": [x0, cy], "body": [x1, cy]},
        {"name": "out", "point": [x1, cy], "body": [x0, cy]},
    ]
    return img, t


def instrument():
    img = _canvas()
    cx, cy, r = S // 2, S // 2, 40
    _circle(img, (cx, cy), r)
    _line(img, (cx - r + 6, cy), (cx + r - 6, cy))  # ISA "field mounted" divider
    t = [{"name": "conn", "point": [cx, cy + r], "body": [cx, cy]}]
    return img, t


def instrument_shared():
    img = _canvas()
    cx, cy, r = S // 2, S // 2, 38
    _poly(img, [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)])
    _circle(img, (cx, cy), r)
    t = [{"name": "conn", "point": [cx, cy + r], "body": [cx, cy]}]
    return img, t


def motor():
    img = _canvas()
    cx, cy, r = S // 2, S // 2, 38
    _circle(img, (cx, cy), r)
    cv2.putText(img, "M", (cx - 14, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (INK, INK, INK), 2)
    t = [{"name": "shaft", "point": [cx, cy + r], "body": [cx, cy]}]
    return img, t


BUILDERS = {
    "Valve_Gate": [gate_valve],
    "Valve_Control": [control_valve],
    "Valve_Check": [check_valve],
    "Pump_Centrifugal": [centrifugal_pump],
    "Vessel": [vessel_vertical, vessel_horizontal],
    "Heat_Exchanger": [heat_exchanger, shell_tube_exchanger],
    "Compressor": [compressor],
    "Filter": [filter_sym],
    "Instrument": [instrument, instrument_shared],
    "Motor": [motor],
}


def main() -> None:
    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)
    total = 0
    for cls, builders in BUILDERS.items():
        d = OUT / cls
        d.mkdir(parents=True, exist_ok=True)
        for fn in builders:
            img, terminals = fn()
            name = fn.__name__
            cv2.imwrite(str(d / f"{name}.png"), img)
            (d / f"{name}.terminals.json").write_text(json.dumps(terminals, indent=2))
            total += 1
    print(f"[pid-ref] wrote {total} symbols across {len(BUILDERS)} classes -> {OUT}")


if __name__ == "__main__":
    main()
