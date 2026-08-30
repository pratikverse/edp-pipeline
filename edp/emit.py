"""Stages 7-8 - JSON netlist and graph output."""
from __future__ import annotations


# ===========================================================================
# json_out.py
# ===========================================================================

from edp.types import DrawingResult


def _connections_for(symbol_id: str, nets) -> list[str]:
    connections: list[str] = []
    for net in nets:
        member_ids = {sid for sid, _tidx in net.terminals}
        if symbol_id not in member_ids:
            continue
        for other_id in sorted(member_ids - {symbol_id}):
            if other_id not in connections:
                connections.append(other_id)
    return connections


def to_json_dict(result: DrawingResult) -> dict:
    return {
        "symbols": [
            {
                "id": symbol.id,
                "type": symbol.type,
                "coordinates": list(symbol.bbox),
                "connections": _connections_for(symbol.id, result.nets),
            }
            for symbol in result.symbols
        ]
    }

# ===========================================================================
# graph_out.py
# ===========================================================================

import math
from pathlib import Path

import networkx as nx

# to_json_dict: defined above
from edp.types import DrawingResult


def build_bipartite_graph(result: DrawingResult) -> nx.Graph:
    # GraphML attributes must be scalar (str/int/float/bool) and non-None,
    # so lists and Nones are stringified here rather than at every call site.
    g = nx.Graph()
    for symbol in result.symbols:
        g.add_node(
            symbol.id,
            bipartite="symbol",
            type=symbol.type,
            value=symbol.value or "",
            bbox=str(list(symbol.bbox)),
            confidence=symbol.confidence,
            label_source=symbol.label_source,
        )
    for net in result.nets:
        g.add_node(
            net.id,
            bipartite="net",
            junction_count=len(net.junctions),
            terminal_count=len(net.terminals),
            confidence=net.confidence,
        )
        for symbol_id, terminal_index in net.terminals:
            g.add_edge(
                symbol_id,
                net.id,
                terminal_index=terminal_index,
            )
    return g


def build_component_graph_from_json(json_dict: dict) -> nx.Graph:
    """The delivered graph: nodes = symbols (typed), edges = `connections`
    pairs, straight from the trimmed JSON — no re-derivation from nets,
    terminals, or anything else internal. `connections` is already
    symmetric, so each pair only needs adding once."""
    g = nx.Graph()
    for symbol in json_dict["symbols"]:
        g.add_node(symbol["id"], type=symbol["type"])
    for symbol in json_dict["symbols"]:
        for other_id in symbol["connections"]:
            g.add_edge(symbol["id"], other_id)
    return g


def export_all(result: DrawingResult, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bipartite = build_bipartite_graph(result)
    component_graph = build_component_graph_from_json(to_json_dict(result))

    graphml_path = out_dir / f"{result.drawing_id}_graph.graphml"
    nx.write_graphml(bipartite, graphml_path)

    json_path = out_dir / f"{result.drawing_id}_graph.json"
    import json as _json

    json_path.write_text(_json.dumps(nx.node_link_data(bipartite, edges="edges"), indent=2), encoding="utf-8")

    png_path = out_dir / f"{result.drawing_id}_graph.png"
    _render_png(component_graph, png_path)

    return {
        "graphml": str(graphml_path),
        "json": str(json_path),
        "png": str(png_path),
    }


def _render_png(graph: nx.Graph, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no symbols detected", ha="center", va="center")
    else:
        pos = _grid_of_components_layout(graph)
        labels = {n: f"{n}\n{d.get('type', '')}" for n, d in graph.nodes(data=True)}
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#5b9dff", width=1.5)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#cfe8ff", node_size=900, edgecolors="#5b9dff")
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _grid_of_components_layout(graph: nx.Graph) -> dict:
    """Lays out each connected component on its own, then tiles the
    components on a grid.

    A single spring_layout call over the whole graph was tried first and
    rejected: connected clusters collapsed tight enough for the node
    circles to fully overlap and hide the edges between them, regardless
    of the global `k` spacing parameter — verified visually, edges existed
    in the graph object but were invisible in the render. Laying out each
    component in its own local coordinate space with a spacing scaled to
    *that* component's size, then placing components on a grid with a
    fixed gap, guarantees no cross-component collision and enough
    intra-component spacing for edges to be visible.
    """
    components = list(nx.connected_components(graph))
    n_components = len(components)
    grid_cols = max(1, math.ceil(math.sqrt(n_components)))
    cell_size = 3.0

    pos: dict = {}
    for idx, component in enumerate(components):
        subgraph = graph.subgraph(component)
        k = 1.0 / math.sqrt(max(len(component), 1))
        local_pos = nx.spring_layout(subgraph, k=k, iterations=200, seed=42)

        row, col = divmod(idx, grid_cols)
        offset_x, offset_y = col * cell_size, -row * cell_size
        for node, (x, y) in local_pos.items():
            pos[node] = (x + offset_x, y + offset_y)
    return pos
