"""Parses .kicad_sym files (S-expression format) and rasterizes each
symbol to a reference PNG with a terminal template, for use by
classify/library.py. See docs 02/06/07 (KiCad as the reference-library
source; D4/D5 are validation inputs only, not training data).

Only the graphic primitives actually present in the fetched Device/Diode/
Transistor/power/Switch libraries are supported: polyline, circle,
rectangle, 3-point arc, and pin. Anything else (text, embedded fonts) is
ignored — it does not affect the rendered shape.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Minimal S-expression parser. No external dependency: the grammar used by
# .kicad_sym files is a small, well-behaved subset (atoms, quoted strings,
# nested parens) that a ~30-line tokenizer/recursive-descent parser handles
# completely, without pulling in a general Lisp-reader dependency.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\(|\)|[^\s()]+')


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _parse(tokens: list[str], pos: int = 0):
    assert tokens[pos] == "("
    pos += 1
    node: list = []
    while tokens[pos] != ")":
        if tokens[pos] == "(":
            child, pos = _parse(tokens, pos)
            node.append(child)
        else:
            tok = tokens[pos]
            if tok.startswith('"') and tok.endswith('"'):
                tok = tok[1:-1].replace('\\"', '"')
            node.append(tok)
            pos += 1
    return node, pos + 1


def parse_sexp(text: str) -> list:
    tokens = _tokenize(text)
    node, _ = _parse(tokens, 0)
    return node


def _find_all(node: list, tag: str) -> list[list]:
    return [child for child in node if isinstance(child, list) and child and child[0] == tag]


def _find_one(node: list, tag: str) -> list | None:
    found = _find_all(node, tag)
    return found[0] if found else None


def _xy(node: list) -> tuple[float, float]:
    return float(node[1]), float(node[2])


# ---------------------------------------------------------------------------
# Graphic model
# ---------------------------------------------------------------------------


@dataclass
class KicadPin:
    x: float
    y: float
    angle: float  # degrees; direction the pin points OUTWARD from the body
    length: float
    number: str


@dataclass
class KicadSymbol:
    name: str
    polylines: list[list[tuple[float, float]]] = field(default_factory=list)
    circles: list[tuple[float, float, float]] = field(default_factory=list)  # cx, cy, r
    rectangles: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0,y0,x1,y1
    arcs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    pins: list[KicadPin] = field(default_factory=list)


def _parse_graphic_subsymbol(sub: list, out: KicadSymbol) -> None:
    for item in sub:
        if not isinstance(item, list) or not item:
            continue
        tag = item[0]
        if tag == "polyline":
            pts_node = _find_one(item, "pts")
            pts = [_xy(p) for p in pts_node[1:] if isinstance(p, list) and p[0] == "xy"]
            out.polylines.append(pts)
        elif tag == "circle":
            center = _xy(_find_one(item, "center"))
            radius = float(_find_one(item, "radius")[1])
            out.circles.append((center[0], center[1], radius))
        elif tag == "rectangle":
            start = _xy(_find_one(item, "start"))
            end = _xy(_find_one(item, "end"))
            out.rectangles.append((start[0], start[1], end[0], end[1]))
        elif tag == "arc":
            start = _xy(_find_one(item, "start"))
            mid = _xy(_find_one(item, "mid"))
            end = _xy(_find_one(item, "end"))
            out.arcs.append((start, mid, end))
        elif tag == "pin":
            at = _find_one(item, "at")
            x, y, angle = float(at[1]), float(at[2]), float(at[3]) if len(at) > 3 else 0.0
            length = float(_find_one(item, "length")[1])
            number_node = _find_one(item, "number")
            number = number_node[1] if number_node else ""
            out.pins.append(KicadPin(x=x, y=y, angle=angle, length=length, number=number))


def load_symbols(path: str | Path) -> list[KicadSymbol]:
    """A .kicad_sym file can define several top-level symbols (rare in our
    curated set, but Device libraries sometimes group variants). Each
    top-level `(symbol "Name" ...)` may itself contain nested
    `(symbol "Name_0_1" ...)` / `(symbol "Name_1_1" ...)` sub-blocks holding
    the actual graphics/pins (KiCad's per-unit convention) — both are
    merged into one KicadSymbol per top-level name."""
    text = Path(path).read_text(encoding="utf-8")
    root = parse_sexp(text)
    results = []
    for top in _find_all(root, "symbol"):
        name = top[1]
        symbol = KicadSymbol(name=name)
        _parse_graphic_subsymbol(top, symbol)
        for nested in _find_all(top, "symbol"):
            _parse_graphic_subsymbol(nested, symbol)
        if symbol.polylines or symbol.circles or symbol.rectangles or symbol.arcs or symbol.pins:
            results.append(symbol)
    return results


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


@dataclass
class RenderedSymbol:
    image: np.ndarray  # HxWx3 uint8, white background, black ink
    # (pin number, tip point, body point) in image space. The tip is where
    # a wire attaches; tip-minus-body gives the pin's outward direction,
    # used for directional terminal snapping (docs/06_data_model.md).
    terminals: list[tuple[str, tuple[int, int], tuple[int, int]]]


def _bounds(symbol: KicadSymbol, pin_reach: bool = True) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for poly in symbol.polylines:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    for cx, cy, r in symbol.circles:
        xs += [cx - r, cx + r]
        ys += [cy - r, cy + r]
    for x0, y0, x1, y1 in symbol.rectangles:
        xs += [x0, x1]
        ys += [y0, y1]
    for start, mid, end in symbol.arcs:
        for x, y in (start, mid, end):
            xs.append(x)
            ys.append(y)
    for pin in symbol.pins:
        xs.append(pin.x)
        ys.append(pin.y)
        if pin_reach:
            rad = math.radians(pin.angle)
            xs.append(pin.x + pin.length * math.cos(rad))
            ys.append(pin.y + pin.length * math.sin(rad))
    if not xs:
        return (-2.54, -2.54, 2.54, 2.54)
    return (min(xs), min(ys), max(xs), max(ys))


def render_symbol(symbol: KicadSymbol, scale: float = 20.0, margin_px: int = 12) -> RenderedSymbol:
    """Rasterizes to a white-background PNG. KiCad's Y axis points up;
    image Y points down, so Y is negated during projection."""
    x0, y0, x1, y1 = _bounds(symbol)
    w = int((x1 - x0) * scale) + 2 * margin_px
    h = int((y1 - y0) * scale) + 2 * margin_px
    w, h = max(w, 20), max(h, 20)

    def proj(x: float, y: float) -> tuple[int, int]:
        px = int((x - x0) * scale) + margin_px
        py = int((y1 - y) * scale) + margin_px  # flip Y
        return px, py

    img = np.full((h, w, 3), 255, dtype=np.uint8)
    thickness = max(1, int(scale * 0.05))

    for poly in symbol.polylines:
        pts = np.array([proj(x, y) for x, y in poly], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)

    for cx, cy, r in symbol.circles:
        center = proj(cx, cy)
        radius = max(1, int(r * scale))
        cv2.circle(img, center, radius, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    for rx0, ry0, rx1, ry1 in symbol.rectangles:
        p0, p1 = proj(rx0, ry0), proj(rx1, ry1)
        cv2.rectangle(img, p0, p1, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    for start, mid, end in symbol.arcs:
        _draw_arc(img, [proj(*start), proj(*mid), proj(*end)], thickness)

    terminals: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for pin in symbol.pins:
        # KiCad's `(at x y angle)` is the pin's OUTER electrical connection
        # point — where a wire attaches in a real schematic — and `length`
        # extends INWARD from there toward the symbol body, in the
        # direction `angle` points. (Confirmed against R_US: pin 1 sits at
        # y=3.81, outside the zigzag's y=2.286 extent, with angle 270 i.e.
        # -y — walking the length moves *toward* y=2.286, into the body.)
        # This was inverted in an earlier version of this function, which
        # silently placed every terminal near the component body instead
        # of at the true wire contact point.
        rad = math.radians(pin.angle)
        inner_x = pin.x + pin.length * math.cos(rad)
        inner_y = pin.y + pin.length * math.sin(rad)
        p_outer, p_inner = proj(pin.x, pin.y), proj(inner_x, inner_y)
        if pin.length > 0:
            cv2.line(img, p_outer, p_inner, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
        # Terminal point = the true wire-contact point (outer). Inner is
        # kept only to derive the pin's outward direction (outer - inner).
        terminals.append((pin.number, p_outer, p_inner))

    return RenderedSymbol(image=img, terminals=terminals)


def _draw_arc(img: np.ndarray, pts_px: list[tuple[int, int]], thickness: int) -> None:
    """Draws a circular arc through 3 image-space points (start, mid, end)
    by fitting the circle through them algebraically, then using
    cv2.ellipse's angle range. Falls back to a straight polyline if the
    three points are (near-)collinear, which fitting a circle to would
    blow up numerically."""
    (x1, y1), (x2, y2), (x3, y3) = pts_px
    a = np.array([[x1 - x2, y1 - y2], [x1 - x3, y1 - y3]], dtype=np.float64)
    if abs(np.linalg.det(a)) < 1e-6:
        pts = np.array(pts_px, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)
        return

    b = np.array(
        [
            (x1**2 - x2**2 + y1**2 - y2**2) / 2,
            (x1**2 - x3**2 + y1**2 - y3**2) / 2,
        ]
    )
    cx, cy = np.linalg.solve(a, b)
    radius = math.hypot(x1 - cx, y1 - cy)

    def ang(x, y):
        return math.degrees(math.atan2(y - cy, x - cx))

    a1, a_mid, a2 = ang(x1, y1), ang(x2, y2), ang(x3, y3)
    # Normalise so the arc sweeps through the mid angle (handles wraparound).
    if not (min(a1, a2) <= a_mid <= max(a1, a2)):
        if a1 > a2:
            a2 += 360
        else:
            a1 += 360
    cv2.ellipse(
        img,
        (int(cx), int(cy)),
        (int(radius), int(radius)),
        0,
        min(a1, a2),
        max(a1, a2),
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )


def render_kicad_file(path: str | Path) -> list[tuple[str, RenderedSymbol]]:
    """Returns [(symbol_name, RenderedSymbol), ...] for every top-level
    symbol defined in the file."""
    return [(s.name, render_symbol(s)) for s in load_symbols(path)]
