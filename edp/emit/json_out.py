"""Nets -> pairwise `connections` -> JSON.

`to_json_dict` emits exactly the shape from the problem statement's own
example — a top-level `symbols` array, each entry with `id`, `type`,
`coordinates`, `connections` and nothing else. Connections reference other
symbols by id only ("R1", "C1", ...), matching the example format.

The richer internal representation (per-net terminal/junction detail,
confidence, rotation, OCR provenance, validation flags) still exists and
still matters — it drives the graph construction in emit/graph_out.py and
the checks in validate/checks.py — it's just not dumped into this file.
See docs/03_json_schema_spec.md.
"""
from __future__ import annotations

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