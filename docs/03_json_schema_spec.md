# JSON Output Schema Spec

Based on the example provided in the problem statement.

## The default output — exactly the example schema, nothing more

The problem statement example carries exactly four fields:

```json
{"symbols": [{"id": "S1", "type": "Circuit Breaker",
              "coordinates": [120, 250, 180, 310], "connections": ["S2", "S5"]}]}
```

`edp run` emits **exactly this shape** — a top-level `symbols` array, each
entry with `id`, `type`, `coordinates`, `connections` and nothing else.
`connections` is a flat list of other symbol ids only (`"R1"`, `"C1"`),
never nested objects. No top-level `metadata`, no `nets` block, no
per-symbol `value`/`confidence`/`terminals`/`rotation`. This was a deliberate
simplification — the richer representation described below was cut back to
this on request, on the reasoning that a smaller, exactly-conformant output
is more useful than a larger one a consumer has to filter down themselves.

`coordinates` ordering is `[x_min, y_min, x_max, y_max]`; this is inferred
from the example, not stated explicitly in the problem statement.

Implemented in `emit/json_out.py::to_json_dict`.

## Where the richer data still lives

Nothing was deleted from the pipeline — only from what gets written to the
primary output file. The fuller internal representation (per-symbol value,
confidence, rotation, terminal points, OCR provenance; per-net terminal and
junction detail; the validation block) still exists and still does real
work:
- It drives graph construction (`emit/graph_out.py` — node/edge attributes
  on the bipartite and projected graphs come from it)
- It drives the self-consistency checks (`validate/checks.py`)
- It's the source for `emit/graph_out.py`'s bipartite graph, which does
  carry this richer detail (net/terminal-level attributes) even though the
  trimmed JSON itself doesn't

See `06_data_model.md` for why a net (an N-way conductor) rather than a
pairwise edge is still the primary connectivity object internally, even
though the emitted JSON flattens it to pairwise `connections` — that
flattening is exactly what `to_json_dict` does at the boundary.

## Symbol object (emitted)
```json
{
  "id": "R1",
  "type": "Resistor",
  "coordinates": [x_min, y_min, x_max, y_max],
  "connections": ["C1", "IC1"]
}
```

- **id**: instance identifier, resolved via OCR association where
  available; falls back to an auto-generated id (`SYM_003`) if no label is
  detected nearby. (Which case applied is tracked internally as
  `label_source` on the `Symbol` object, but is not part of the emitted
  JSON.)
- **type**: symbol class from the reference embedding library (e.g.
  Resistor, Capacitor, BJT_NPN, MOSFET_N, Ground, Crystal, Inductor,
  Transformer, Switch, Antenna, Diode, LED, Zener, Battery, Fuse) — see
  `data/reference/` for the full current set, sourced from KiCad
  (`05_open_questions_and_assumptions.md`).
- **coordinates**: bounding box in image pixel space.
- **connections**: flat list of connected symbol ids, undirected — two
  symbols appear in each other's list iff they share a net. Matches the
  problem statement's example format directly.

## Open schema questions (see 05_open_questions_and_assumptions.md)
- Whether ground symbols should be modeled as separate instances or merged
  into a shared logical node.
- Whether `connections` should be directional or purely undirected (current
  assumption: undirected, since these are passive schematic connections).
