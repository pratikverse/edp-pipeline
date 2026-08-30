"""Stages 5-6 - wire skeletonization, junction-dot detection, net construction."""
from __future__ import annotations


# ===========================================================================
# skeleton.py
# ===========================================================================

import cv2
import numpy as np
from skimage.morphology import skeletonize

from edp.types import Symbol, Terminal

_MIN_RUN = 11  # px; a straight ink run at least this long is a candidate wire
_STUB = 6      # px; how far outside a bbox to check a run continues


def subtract_symbols(binary: np.ndarray, symbols: list[Symbol], pad: int = 2) -> np.ndarray:
    """Removes symbol ink so only wire pixels remain — but keeps wires that
    merely *pass through* a symbol's bounding box.

    The old version zeroed the whole bbox rectangle. A wide symbol (D4's
    MOSFET box is 148x180 px) then erased any wire routed near or behind
    it, silently severing nets (docs/08 section 2.6). Here each bbox is
    still cleared, then any long straight horizontal/vertical run that
    entered one side of the box and left the opposite side — i.e. a wire
    crossing, not the symbol's own body — is restored.
    """
    h, w = binary.shape[:2]
    out = binary.copy()

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (_MIN_RUN, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _MIN_RUN))
    h_runs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    v_runs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    for s in symbols:
        bx0, by0, bx1, by1 = s.bbox
        xa, ya = max(0, bx0 - pad), max(0, by0 - pad)
        xb, yb = min(w, bx1 + pad), min(h, by1 + pad)
        if xb <= xa or yb <= ya:
            continue

        out[ya:yb, xa:xb] = 0

        left_col = binary[ya:yb, max(0, xa - _STUB):xa]
        right_col = binary[ya:yb, xb:min(w, xb + _STUB)]
        rows_cross = left_col.any(axis=1) & right_col.any(axis=1)
        for i, crossing in enumerate(rows_cross):
            if crossing:
                r = ya + i
                out[r, xa:xb] = np.maximum(out[r, xa:xb], h_runs[r, xa:xb])

        top_row = binary[max(0, ya - _STUB):ya, xa:xb]
        bot_row = binary[yb:min(h, yb + _STUB), xa:xb]
        cols_cross = top_row.any(axis=0) & bot_row.any(axis=0)
        for j, crossing in enumerate(cols_cross):
            if crossing:
                c = xa + j
                out[ya:yb, c] = np.maximum(out[ya:yb, c], v_runs[ya:yb, c])

    return out


def skeletonize_wires(wire_mask: np.ndarray) -> np.ndarray:
    """Morphological thinning to 1px-wide skeleton. Returns uint8 0/255."""
    bool_mask = wire_mask > 0
    thin = skeletonize(bool_mask)
    return (thin.astype(np.uint8)) * 255


def refine_terminals(
    symbols: list[Symbol], binary: np.ndarray, band: int = 14, min_run: int = 2
) -> list[Symbol]:
    """Move each symbol's terminals onto the points where wires actually
    touch its bounding box, instead of trusting the KiCad template scaled
    blindly onto the box.

    The template gives the expected terminal *count* and *pin direction*;
    the box bounds the drawn symbol, but KiCad's normalised pin fractions
    are relative to a render that also includes the outward pin leads, so
    the scaled points land inside the body or past the wire end
    (docs/08 section 2.5 — this is why terminal_snap_radius had to be
    loosened 12 -> 25 -> 30 px just to catch anything).

    Here: find the ink runs crossing a thin band just outside each of the
    four box edges (a wire connecting to the symbol), match each to the
    nearest template terminal, and snap that terminal to the run's
    midpoint with the edge's outward normal as its direction. Template
    terminals with no run nearby are left untouched (fallback); runs with
    no matching template terminal become extra inferred terminals (a
    4-pin optocoupler matched to a 2-pin template really does have 4
    connections).
    """
    h, w = binary.shape[:2]
    for s in symbols:
        x0, y0, x1, y1 = s.bbox
        attach: list[tuple[int, int, float]] = []  # (x, y, outward_dir_deg)

        # top / bottom edges: scan a horizontal strip, project onto x
        for edge_y, y_a, y_b, out_dir in (
            (y0, max(0, y0 - band), y0, 270.0),
            (y1, y1, min(h, y1 + band), 90.0),
        ):
            if y_b <= y_a:
                continue
            col_has = (binary[y_a:y_b, x0:x1] > 0).any(axis=0)
            for c0, c1 in _runs(col_has, min_run):
                attach.append((x0 + (c0 + c1) // 2, edge_y, out_dir))

        # left / right edges: scan a vertical strip, project onto y
        for edge_x, x_a, x_b, out_dir in (
            (x0, max(0, x0 - band), x0, 180.0),
            (x1, x1, min(w, x1 + band), 0.0),
        ):
            if x_b <= x_a:
                continue
            row_has = (binary[y0:y1, x_a:x_b] > 0).any(axis=1)
            for r0, r1 in _runs(row_has, min_run):
                attach.append((edge_x, y0 + (r0 + r1) // 2, out_dir))

        if not attach:
            continue

        used: set[int] = set()
        for term in s.terminals:
            tx, ty = term.point
            order = sorted(
                range(len(attach)),
                key=lambda i: (attach[i][0] - tx) ** 2 + (attach[i][1] - ty) ** 2,
            )
            for i in order:
                if i in used:
                    continue
                used.add(i)
                px, py, out_dir = attach[i]
                term.point = (px, py)
                term.direction_deg = out_dir
                term.source = "inferred"
                break

        for i, (px, py, out_dir) in enumerate(attach):
            if i in used:
                continue
            s.terminals.append(
                Terminal(
                    symbol_id=s.id,
                    index=len(s.terminals),
                    point=(px, py),
                    source="inferred",
                    direction_deg=out_dir,
                )
            )
    return symbols


def _runs(mask_1d: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[(start, end)] index ranges of contiguous True runs at least min_len long."""
    out = []
    idx = np.flatnonzero(mask_1d)
    if len(idx) == 0:
        return out
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    for s, e in zip(starts, ends):
        if idx[e] - idx[s] + 1 >= min_len:
            out.append((int(idx[s]), int(idx[e])))
    return out

# ===========================================================================
# junctions.py
# ===========================================================================

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from edp.config import WiresConfig
from edp.types import Point

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def detect_junction_dots(binary: np.ndarray, cfg: WiresConfig) -> list[Point]:
    """Finds filled dots by local fill-ratio at skeleton branch points,
    not by contour shape.

    A `cv2.findContours` approach was tried first and rejected: a dot
    sitting directly on a wire (the normal case — a dot only exists where
    wires meet) is 8-connected to that wire, so the whole dot+wire run
    becomes one contour, and `minEnclosingCircle` of that shape is nothing
    like a small circle. Verified this was silently dropping most real
    dots (only ones forming their own isolated contour survived — 30 out
    of many more actually present in D5). Checking local ink density in a
    small disk around every skeleton branch point instead is invariant to
    what else that point happens to be connected to: a plain crossing
    fills only the width of the two crossing strokes, a drawn dot fills
    most of the disk.
    """
    skeleton = skeletonize(binary > 0).astype(np.uint8)
    degree = convolve(skeleton, _NEIGHBOR_KERNEL, mode="constant", cval=0) * skeleton
    branch_ys, branch_xs = np.where(degree >= 3)

    # radius=max_radius diluted fill ratio far below threshold (a ~2px dot
    # fills only ~16% of a radius-5 disk); radius=min_radius overfired on
    # ordinary zigzag/coil corners, which are locally dense enough to pass
    # a tiny disk's fill check too. max_radius-1 was the empirical sweet
    # spot verified by eye against D5: catches dots at real junctions
    # without lighting up every symbol corner.
    radius = cfg.junction_dot_max_radius - 1
    disk_area = np.pi * radius * radius
    h, w = binary.shape[:2]

    dots: list[Point] = []
    for x, y in zip(branch_xs.tolist(), branch_ys.tolist()):
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(w, x + radius + 1), min(h, y + radius + 1)
        disk_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(disk_mask, (x - x0, y - y0), radius, 1, thickness=-1)
        region = binary[y0:y1, x0:x1] > 0
        fill_ratio = float(np.count_nonzero(region & (disk_mask > 0))) / disk_area
        if fill_ratio > cfg.junction_dot_fill_ratio:
            dots.append((x, y))
    return dots


def is_dotted_crossing(point: Point, dots: list[Point], tolerance: int = 4) -> bool:
    px, py = point
    return any(abs(px - dx) <= tolerance and abs(py - dy) <= tolerance for dx, dy in dots)

# ===========================================================================
# nets.py
# ===========================================================================

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from edp.config import WiresConfig
from edp.types import Net, Point, Symbol

# is_dotted_crossing: defined above

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _pixel_graph(skeleton: np.ndarray) -> nx.Graph:
    """8-connected graph over skeleton foreground pixels. Node = (row, col)."""
    ys, xs = np.where(skeleton > 0)
    pixel_set = set(zip(ys.tolist(), xs.tolist()))
    g = nx.Graph()
    g.add_nodes_from(pixel_set)
    for y, x in pixel_set:
        for dy, dx in _NEIGHBOR_OFFSETS:
            n = (y + dy, x + dx)
            if n in pixel_set:
                g.add_edge((y, x), n)
    return g


@dataclass
class _CrossingRegion:
    pixels: set
    arms: list  # list of (arm_pixel, direction_vector, outside_neighbor_pixel)


def _find_crossing_regions(g: nx.Graph) -> list[_CrossingRegion]:
    """Branch pixels (degree >= 3) merged into contiguous regions — a real
    crossing can span more than one skeleton pixel."""
    branch_nodes = {n for n in g.nodes if g.degree[n] >= 3}
    if not branch_nodes:
        return []

    branch_subgraph = g.subgraph(branch_nodes)
    regions: list[_CrossingRegion] = []
    for component in nx.connected_components(branch_subgraph):
        arms = []
        for node in component:
            for neighbor in g.neighbors(node):
                if neighbor in component:
                    continue
                dy = neighbor[0] - node[0]
                dx = neighbor[1] - node[1]
                arms.append((node, (dy, dx), neighbor))
        regions.append(_CrossingRegion(pixels=set(component), arms=arms))
    return regions


def _pair_by_opposite_angle(arms: list) -> list[tuple[int, int]]:
    """Given >=4 arms, pair each with the one closest to 180 degrees away
    (a straight through-path), leaving the remainder paired by exclusion.
    Returns index pairs into `arms`."""
    n = len(arms)
    angles = [math.atan2(dy, dx) for (_node, (dy, dx), _out) in arms]
    used = set()
    pairs: list[tuple[int, int]] = []
    order = sorted(range(n), key=lambda i: angles[i])
    for i in order:
        if i in used:
            continue
        best_j, best_score = None, -1.0
        for j in range(n):
            if j == i or j in used:
                continue
            diff = abs(angles[i] - angles[j])
            diff = min(diff, 2 * math.pi - diff)
            score = diff  # closer to pi (opposite direction) is better
            if score > best_score:
                best_score, best_j = score, j
        if best_j is not None:
            pairs.append((i, best_j))
            used.add(i)
            used.add(best_j)
    return pairs


def build_nets(
    skeleton: np.ndarray,
    dots: list[Point],
    symbols: list[Symbol],
    cfg: WiresConfig,
) -> list[Net]:
    g = _pixel_graph(skeleton)
    regions = _find_crossing_regions(g)

    # Rewire graph: for each undotted 4-arm crossing, remove the region's
    # pixels and directly connect opposite-angle arm neighbors instead —
    # this is the structural split described in docs/06_data_model.md.
    rewired = g.copy()
    for region in regions:
        centroid = (
            sum(p[0] for p in region.pixels) / len(region.pixels),
            sum(p[1] for p in region.pixels) / len(region.pixels),
        )
        dotted = is_dotted_crossing((int(centroid[1]), int(centroid[0])), dots)

        rewired.remove_nodes_from(region.pixels)

        if dotted or len(region.arms) < 4:
            # Keep connected: re-add a single synthetic junction node joining
            # all arms. (T-junctions without a dot are not a valid drawing
            # convention here; treated as connected defensively rather than
            # silently dropped — see docs/05 limitations.)
            junction_node = (int(centroid[0]), int(centroid[1]), "junction")
            rewired.add_node(junction_node)
            for _node, _dir, outside in region.arms:
                rewired.add_edge(junction_node, outside)
        else:
            # Undotted crossing: split into independent pass-through pairs.
            for i, j in _pair_by_opposite_angle(region.arms):
                _, _, outside_i = region.arms[i]
                _, _, outside_j = region.arms[j]
                rewired.add_edge(outside_i, outside_j)

    nets: list[Net] = []
    for idx, component in enumerate(nx.connected_components(rewired)):
        if len(component) < 2:
            continue
        sub = rewired.subgraph(component)
        pixel_nodes = [n for n in component if len(n) == 2]
        junction_nodes = [n for n in component if len(n) == 3]
        polyline = [[(int(x), int(y)) for (y, x) in pixel_nodes]]
        junctions_in_net = [(int(n[1]), int(n[0])) for n in junction_nodes]
        nets.append(
            Net(
                id=f"N{idx:03d}",
                terminals=[],
                polyline=polyline,
                junctions=junctions_in_net,
                confidence=1.0,
            )
        )

    _attach_terminals(nets, rewired, symbols, cfg)
    return nets


def _attach_terminals(nets: list[Net], rewired: nx.Graph, symbols: list[Symbol], cfg: WiresConfig) -> None:
    """Snaps each symbol terminal to whichever net has skeleton pixels
    nearby.

    When the terminal has a known pin direction (KiCad-sourced, see
    docs/06_data_model.md), snapping searches a *cone* extending outward
    from the terminal in that direction rather than a blind isotropic
    circle: a wire almost always leaves a pin travelling along the pin's
    own axis, so restricting the search direction lets the reach extend
    further (terminal_directional_reach) without the false-positive risk
    a correspondingly larger plain radius would carry in dense areas.
    Terminals with no known direction (inferred, or a library entry with
    no terminal template) fall back to the isotropic search.
    """
    if not nets:
        return

    components = list(nx.connected_components(rewired))
    comp_for_net = {}
    for net, component in zip(nets, [c for c in components if len(c) >= 2]):
        comp_for_net[net.id] = component

    for symbol in symbols:
        for terminal in symbol.terminals:
            best_net = _snap_terminal(terminal, nets, comp_for_net, cfg)
            if best_net is not None:
                best_net.terminals.append((symbol.id, terminal.index))
                terminal.net_id = best_net.id


def _snap_terminal(terminal, nets: list[Net], comp_for_net: dict, cfg: WiresConfig) -> Net | None:
    tx, ty = terminal.point
    directional = terminal.direction_deg is not None
    max_dist = cfg.terminal_directional_reach if directional else cfg.terminal_snap_radius
    half_cone = cfg.terminal_snap_cone_deg / 2

    best_net, best_dist = None, max_dist + 1
    for net in nets:
        component = comp_for_net.get(net.id, set())
        for node in component:
            py, px = node[0], node[1]
            dx, dy = px - tx, py - ty
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= best_dist:
                continue
            if directional and dist > 0:
                angle_diff = abs(_angle_diff_deg(terminal.direction_deg, math.degrees(math.atan2(dy, dx))))
                if angle_diff > half_cone:
                    continue
            best_dist, best_net = dist, net
    return best_net


def _angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b) % 360
    return diff if diff <= 180 else 360 - diff
