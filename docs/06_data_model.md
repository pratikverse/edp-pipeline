# Core Data Model — Terminals, Nets, and the Graph

This doc fills the gap between "trace the wires" (stage 5) and
`"connections": ["C1", "IC1"]` (stage 8). It defines the intermediate
objects the pipeline actually operates on, and why they exist.

## The problem with going straight to pairwise connections
A wire in a schematic is not a pair — it is an **N-way net**. In D5 the top
rail touches C1, R1 and the IC in a single conductor. If the pipeline emits
pairwise links directly from traced segments, it has to re-derive "these
three are all the same conductor" during dedupe, and every junction merge
becomes a special case. Modelling the net explicitly and expanding to pairs
only at JSON-emit time keeps stage 6 simple and makes stage 7 validation
meaningful.

## Objects

### Terminal
A connection point on a symbol. Produced by the classification stage, not
the wire stage.

```python
Terminal(
    symbol_id: str,        # "T2"
    index: int,            # stable ordinal within the symbol
    name: str | None,      # "B" | "C" | "E" | "1" | "G" | None
    point: (int, int),     # image-space pixel location
    source: str,           # "library" | "inferred" | "ocr"
)
```

Terminals come from the **reference library**: each reference symbol is
stored with its terminal offsets in normalised crop coordinates, which are
transformed onto the matched candidate's bbox at classification time. Where
no library terminals exist (unknown symbols), fall back to `inferred` —
points where wire skeleton pixels meet the bbox boundary.

**Why this matters on the given data:** D4's IC1 has pins 1, 2, 4, 5
explicitly numbered; T4 is labelled G/D/S; every BJT has three legs. Saying
"R4 connects to T2" without naming the leg discards the part of the answer
that carries the circuit meaning. Pin-level connectivity is the difference
between a netlist and an adjacency blob.

### Net
An equivalence class of terminals joined by one continuous conductor.

```python
Net(
    id: str,                       # "N007"
    terminals: list[TerminalRef],
    polyline: list[list[Point]],   # traced skeleton geometry, for overlay/debug
    junctions: list[Point],        # confirmed dot locations on this net
    confidence: float,
)
```

### Connection (derived, not primary)
Emitted only during JSON generation, by expanding each net over its member
terminals. Two symbols are connected iff they share a net.

## How nets are built (stage 6, restated)

1. **Skeletonise** the wire layer (symbol regions subtracted).
2. **Build a skeleton graph**: pixels → nodes at endpoints and branch
   points, edges = traced paths between them.
3. **Decompose crossings.** Every degree-4 node is a decision point:
   - filled dot present → keep as a single junction node (all four
     branches share a net)
   - no dot → **split the node into two independent pass-through paths**
     (the horizontal pair and the vertical pair), so the two conductors
     stay in different connected components.

   Doing the split here, structurally, is what makes the "crossing ≠
   connection" rule automatic rather than a post-hoc correction. This is
   the single highest-leverage step in the whole connectivity stage.
4. **Connected components** of the decomposed graph = candidate nets.
5. **Attach terminals**: snap each symbol terminal to a net whose geometry
   passes within `snap_radius` px. Unattached terminals and nets with fewer
   than two terminals are flagged for stage 7, not silently dropped.

## Graph representation

Build the graph **bipartite** internally — nodes are symbols *and* nets,
edges are terminal attachments:

```
(R1) --pin:1--> [N007] <--pin:2-- (C1)
                   ^
                   |pin:5
                 (IC1)
```

**Why bipartite rather than symbol-to-symbol:** a 5-terminal net expands to
a 10-edge clique in a plain component graph, which visually and
structurally implies pairwise wires that do not exist. The bipartite form
is the faithful representation of the circuit, and it is lossless.

Project it down to the **component graph** (nodes = symbols, edges =
"shares a net") for the delivered artefact, since that matches the
`connections` field the problem statement asks for. Both are exported; the
projection is one NetworkX call.

### Attribute schema
| | attributes |
|---|---|
| symbol node | `type`, `value`, `bbox`, `confidence`, `label_source` |
| net node | `junction_count`, `confidence`, `terminal_count` |
| edge | `terminal_name`, `terminal_index`, `snap_distance` |

### Exports
- `graph.graphml` — full bipartite graph, attributes preserved, opens in Gephi/yEd
- `graph.json` — node-link JSON, for programmatic consumers
- `graph.png` — Graphviz layout of the projected component graph, for the demo

GraphML is the primary export: it is a standard, it round-trips through
NetworkX without loss, and it does not require the reader to run our code.
