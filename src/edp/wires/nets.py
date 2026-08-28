"""Skeleton -> Net objects. This is the core of docs/06_data_model.md: a
degree-4 crossing with no junction dot is split into two independent
pass-through paths *before* connected components are taken, so "crossing
without a dot is not a connection" is structural, not a post-hoc filter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from edp.config import WiresConfig
from edp.types import Net, Point, Symbol

from .junctions import is_dotted_crossing

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
