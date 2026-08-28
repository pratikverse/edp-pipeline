"""Nets -> pairwise `connections` -> JSON.

`to_json_dict` emits exactly the shape from the problem statement's own
example — a top-level `symbols` array, each entry with `id`, `type`,
`coordinates`, `connections` and nothing else. Connections reference other
symbols by id only ("R1", "C1", ...), matching the example format.

The richer internal representation (per-net terminal/junction detail,
confidence, rotation, OCR provenance, validation flags) still exists and
still matters — it drives the graph construction in emit/graph_out.py and
the checks in validate/checks.py — it's just not dumped into this file.
`to_json_dict_extended` below is that fuller view, kept for debugging and
not wired into the CLI/API by default. See docs/03_json_schema_spec.md.
"""
from __future__ import annotations

import datetime

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


def to_json_dict_extended(result: DrawingResult) -> dict:
    """Debug/internal view with everything to_json_dict omits: per-symbol
    value/confidence/rotation/terminals/OCR provenance, per-net terminal
    and junction detail, and the validation block. Not used by the CLI or
    the demo API by default."""
    symbols_out = []
    for symbol in result.symbols:
        connections = _connections_for(symbol.id, result.nets)
        details = []
        for net in result.nets:
            members = [t for t in net.terminals if t[0] == symbol.id]
            if not members:
                continue
            for _sid, tidx in members:
                for oid, otidx in net.terminals:
                    if oid == symbol.id:
                        continue
                    details.append(
                        {
                            "to": oid,
                            "via_net": net.id,
                            "from_terminal": str(tidx),
                            "to_terminal": str(otidx),
                            "junction_type": "dot" if net.junctions else "none",
                        }
                    )
        symbols_out.append(
            {
                "id": symbol.id,
                "type": symbol.type,
                "value": symbol.value,
                "coordinates": list(symbol.bbox),
                "confidence": round(symbol.confidence, 4),
                "connections": connections,
                "connection_details": details,
                "terminals": [
                    {"index": t.index, "name": t.name, "point": list(t.point), "net": t.net_id}
                    for t in symbol.terminals
                ],
                "rotation": symbol.rotation,
                "metadata": {
                    "source": symbol.label_source,
                    "ocr_text_raw": symbol.ocr_text_raw,
                },
            }
        )

    nets_out = [
        {
            "id": net.id,
            "terminals": [{"symbol": sid, "terminal": tidx} for sid, tidx in net.terminals],
            "junction_count": len(net.junctions),
            "confidence": round(net.confidence, 4),
            "polyline": net.polyline,
        }
        for net in result.nets
    ]

    v = result.validation
    validation_out = {
        "unattached_terminals": [{"symbol": s, "terminal": t} for s, t in v.unattached_terminals],
        "isolated_symbols": v.isolated_symbols,
        "single_terminal_nets": v.single_terminal_nets,
        "low_confidence_symbols": [{"id": s, "confidence": round(c, 4)} for s, c in v.low_confidence_symbols],
    }

    return {
        "drawing_id": result.drawing_id,
        "symbols": symbols_out,
        "nets": nets_out,
        "validation": validation_out,
        "metadata": {
            "source_file": result.source_file,
            "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pipeline_version": result.pipeline_version,
        },
    }
