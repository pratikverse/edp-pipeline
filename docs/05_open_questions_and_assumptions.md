# Open Questions & Assumptions

Track decisions made without explicit confirmation from the problem
statement, so they can be revisited or called out explicitly in the final
technical documentation.

## Resolved by inspecting the provided data
3. ~~**Drawing set scope** — how many drawings, what format?~~
   **Resolved:** two drawings, both raster PNG (D4 1002×501 RGBA, D5 680×484
   RGB). No vector/PDF input exists, so there is no embedded-text shortcut —
   OCR is the only text path, and the pipeline commits to it.
4. ~~**Reference symbol library** — standard set or drawing-derived?~~
   **Resolved differently from the original plan:** sourced entirely from
   KiCad's official `Device`/power/Switch/Transistor symbol libraries
   (CC-BY-SA 4.0, `gitlab.com/kicad/libraries/kicad-symbols`), not cropped
   from D4/D5. D4/D5 are held out as validation-only inputs — the pipeline
   is never tuned against ground truth taken from the same drawings it is
   scored on. A custom S-expression parser + rasterizer
   (`edp.classify.kicad_import`) renders each `.kicad_sym` symbol to a PNG
   and extracts its pin geometry as a terminal template in the same step —
   see docs/06_data_model.md and docs/02 for why terminal templates matter.
   Trade-off accepted: KiCad's line weights/proportions differ from D4/D5's
   own rendering, a real domain gap; classification confidence on D4/D5 is
   the read on whether it holds up, not a tuning target.

## Open questions
1. **Ground symbol modeling** — should multiple ground symbols in one
   drawing be treated as separate isolated nodes, or merged into one
   logical "GND" reference node? Current assumption: keep as separate
   instances (matches literal drawing content); note the alternative in
   docs as a configurable post-processing option. Relevant on D5, which has
   several grounds.
2. **Directionality of connections** — are `connections` expected to be
   undirected (symmetric list membership) or should current flow direction
   be inferred? Current assumption: undirected, since nothing in the
   problem statement asks for signal-flow direction. Note that polarised
   parts (D1, ZD1, the batteries, C1 in D4) do carry intrinsic orientation;
   that is recorded as symbol metadata rather than as edge direction.
5. **Multi-terminal granularity in the delivered `connections` field** —
   the example schema is a flat list of symbol ids, which cannot express
   "R4 connects to the base of T2". Current assumption: keep `connections`
   flat and example-conformant, and carry pin-level detail in the additive
   `terminals` / `connection_details` fields (see 03_json_schema_spec.md).
6. **Non-component blocks** — D4 contains a filled "LOAD" block and D5 a
   dashed "SHIELD" boundary. Current assumption: LOAD is a component node
   (it is electrically in-circuit); SHIELD is an annotation region, not a
   node, but its enclosure is retained as metadata on the symbols inside it.
7. **Composite/hierarchical symbols** — IC1 in D4 is an optocoupler drawn as
   a box containing an LED and a phototransistor. Current assumption: treat
   the outer box as a single symbol with numbered pins and do not decompose
   the internals. Decomposition is a possible future extension.

## Bugs found during implementation worth recording
- **KiCad pin geometry was inverted in the first implementation of the
  renderer/importer.** `.kicad_sym`'s `(at x y angle)` is the pin's *outer*
  electrical connection point — where a wire attaches in a real schematic
  — and `length` walks *inward* from there toward the symbol body. The
  first version of `classify/kicad_import.py` treated `(at ...)` as the
  inner anchor and the length-extended point as the outer tip: the
  opposite of KiCad's actual convention. Every terminal was consequently
  placed near the component body rather than at the true wire-contact
  point, which is why terminal snapping initially needed a implausibly
  large radius to find any connections at all, and why an added
  directional search (cone around the pin's outward direction) made
  results *worse* — it was searching into the component, not toward the
  wire. Confirmed by checking the fetched `R_US.kicad_sym`: pin 1 sits at
  y=3.81, outside the zigzag body's y=2.286 extent, with `length` walking
  toward smaller y — i.e. into the body. Fixed by swapping the two
  points; symbols-with-a-connection went from 8/25 to 12/25 on D5 and
  14/30 to 22/30 on D4 in the same run, with no other change. Kept here
  rather than silently corrected, since "why a specific technique/format
  was interpreted a certain way" is exactly what the problem statement
  asks to be able to justify.

## Assumptions carried into the architecture
- Drawings are digitally generated line art (clean, high-contrast), not
  noisy scans or hand-drawn sketches — classical CV stages are designed
  around this. **Confirmed** against both provided files.
- Symbols follow a broadly standardized visual vocabulary (IEC 60617-like),
  making embedding-based nearest-neighbor classification viable without
  fine-tuning — but with **style variation between drawings**, so the
  library holds several exemplars per class.
- Symbols appear at arbitrary 90° orientations, so the reference library is
  rotation-augmented rather than assuming upright placement.
- Wire routing is predominantly orthogonal. Diagonal and curved runs exist
  (D4 has a diagonal MOSFET gate lead) and are handled, but less reliably.
- Colour, where present, is an opportunistic signal only. D5 uses blue and
  black strokes; D4 is monochrome. No stage may require colour to be present.
- Accuracy on any single drawing matters less than the pipeline being
  explainable and generalizable, per the evaluation criteria explicit
  weighting.
- **No labelled ground truth exists.** Two drawings, no annotations. No
  precision/recall figure will be reported; validation is self-consistency
  checks plus visual review, and this is stated as a limitation rather than
  papered over with a hand-scored number.
