# Architecture Overview — Electrical Drawing Interpretation Pipeline

## Objective
Given raster electrical drawings, produce:
1. A structured JSON of detected symbols (id, type, bbox, connections, metadata)
2. A graph representation of the circuit (nodes = components, edges = wires)

## Design principle
Evaluation rewards a **generic, scalable, well-justified** pipeline over raw accuracy.
Every stage below is chosen to (a) require no/minimal labeled training data, and
(b) generalize to new symbol types by adding reference data, not retraining models.

## Input characteristics (from the provided data)
Both drawings are clean, digitally-generated raster line art — the assumption
the classical CV stages are built on holds.

| | D4.png | D5.png |
|---|---|---|
| size | 1002 × 501 | 680 × 484 |
| colour | black on white | **black + blue strokes** |
| content | opto-isolated switch driver | RF oscillator |
| symbols | IC1 (MCT2E, solid box), T1–T3 BJTs **in circles**, T4 MOSFET, ZD1 zener, D1 diode, batteries, S1 switch, LOAD block, R1–R6/R51 | R1–R6, C1–C8, Q1 BJT **bare, no circle**, XTAL, T1 transformer, L2–L5 inductors, IC box, dashed SHIELD box, grounds, antenna |

Three consequences drive the design below:
- **Style varies between drawings** (circled vs. bare transistors). This is
  precisely the generalization case the evaluation probes, and it argues for
  reference-library matching over a fine-tuned detector.
- **Symbols appear in multiple orientations** (R1 vertical, R4 horizontal;
  inductors both ways). Rotation handling is mandatory, not optional.
- **Resolution is low.** At 680 px wide, `XTAL149.89MHz` and `C7 180pF` are a
  few pixels tall. OCR needs upscaling; this is the dominant text-stage risk,
  ahead of noise.

## Pipeline stages

```
Raw drawing (PNG)
        │
        ▼
[1] Preprocessing
    - deskew, binarize, denoise
    - strip title block / border / legend (if present)
    - colour-layer split: D5 draws some conductors in blue. Separating the
      colour channels gives a near-free net-membership prior, used as a
      cross-check in stage 6 rather than a hard dependency (D4 is monochrome,
      so nothing may rely on colour being present).
    - tile if drawing is very large
        │
        ▼
[2] Symbol Localization (candidate proposal)
    - skeleton branch/endpoint density, not thick-vs-thin morphology as
      originally planned: verified empirically that D4/D5 draw symbols and
      wires with the *same* stroke width, so erosion-based thick/thin
      separation either erases everything or barely separates anything.
      What actually distinguishes a symbol is local complexity — several
      skeleton corners/branches packed into a small area, vs. a wire's
      long run of straight degree-2 pixels
    - OCR runs *before* this stage specifically so its token boxes can be
      stripped first: text glyphs have as many skeleton corners as a real
      symbol and would otherwise become false candidates
    - connected components of the density mask, area-filtered, become
      candidate bounding boxes
    - known limitation: dashed-boundary regions (e.g. D5's SHIELD box)
      have sparse geometry and sit right at the density threshold —
      unreliable, tracked in docs/05
        │
        ▼
[3] Symbol Classification (ViT embeddings)
    - crop each candidate bbox
    - embed via pretrained ViT (DINOv2) — no fine-tuning required
    - nearest-neighbor match against a small labeled reference
      embedding library (built from standard symbol sets, e.g. IEC 60617,
      plus any examples confirmed from the provided drawings)
    - reference library is **rotation-augmented** (0/90/180/270, plus
      mirroring): DINOv2 embeddings are not rotation-invariant, and the
      drawings contain the same symbol at multiple orientations. The matched
      rotation is retained — it is what orients the terminal template in
      stage 6.
    - library holds **multiple style variants per class** (circled and bare
      transistors, etc.) since the two drawings already disagree on convention
    - similarity threshold → flag low-confidence / "unknown" symbols
    - attach terminals: transform the reference symbol terminal offsets onto
      the matched bbox (see 06_data_model.md)
        │
        ▼
[4] Text Association (OCR)
    - upscale 3–4× (Lanczos) before OCR — label glyphs are only a few pixels
      tall at native resolution and Tesseract fails on them outright
    - multi-orientation passes (0/90/270): D4 sets "LOAD" vertically
    - associate each text token to nearest symbol bbox
    - resolves instance identity (R1 vs R2) and value (10K, 180pF)
      which geometry/embeddings alone cannot distinguish
    - split combined tokens: "R1 10K" → id + value; "XTAL149.89MHz" → type + value
        │
        ▼
[5] Wire / Line Detection
    - subtract symbol regions from binarized image
    - skeletonize remaining pixels (morphological thinning)
    - build a skeleton graph: nodes at endpoints and branch points,
      edges = traced paths between them
        │
        ▼
[6] Connectivity Inference  → nets, not pairs
    - junction disambiguation: filled dot at intersection = connected,
      plain crossing = not connected (small-circle template match)
    - **crossing decomposition**: a degree-4 skeleton node with no dot is
      structurally split into two independent pass-through paths, so the two
      conductors land in different components. This makes "crossing ≠
      connection" automatic rather than a post-hoc correction.
    - connected components of the decomposed graph = candidate **nets**
    - snap symbol terminals to nets within `terminal_snap_radius`
    - a net is an N-way object; pairwise `connections` are derived from it at
      stage 8, not built here (see 06_data_model.md)
        │
        ▼
[7] Post-processing & Validation
    - dedupe/merge redundant connections
    - sanity checks: unattached terminals, nets with <2 terminals, isolated
      symbols, dangling wire endpoints, asymmetric connection lists
    - confidence scoring per symbol/net
    - all checks are **self-consistency** checks: with two unlabeled drawings
      there is no ground truth to score against (see 07_project_layout.md)
        │
        ▼
[8] JSON Generation
    - expand nets → pairwise `connections`
    - assemble per-schema output (see 03_json_schema_spec.md)
        │
        ▼
[9] Graph Construction
    - build bipartite NetworkX graph (symbol nodes + net nodes)
    - project to component graph for the delivered artefact
    - export GraphML / node-link JSON / rendered PNG
```

## Why this decomposition
- **Localization and classification are decoupled.** Embeddings alone cannot
  localize; classical CV proposals need no training data. This split is what
  makes the pipeline "generic" — new symbol types are handled by adding
  reference embeddings, not retraining a detector.
- **Wires are handled with classical CV, not ML.** Electrical drawings are
  clean line art; skeletonization/Hough is fast, deterministic, and needs
  no training data. ML wire-detection would add complexity without a clear
  accuracy benefit on this input type.
- **OCR is a separate signal from classification.** Geometry tells you
  "this is a resistor"; text tells you "this is R1, 10K". Conflating them
  would be brittle.
- **Nets are first-class; pairwise connections are derived.** A conductor
  touches N terminals, not two. Modelling that directly keeps junction
  merging simple and makes validation meaningful.

## Known limitations (to expand in final documentation)
- Dashed-boundary symbols (relays, shields) need explicit handling.
- Junction-dot detection is the main connectivity failure point on dense
  drawings; at D5 resolution a dot is 3–5 px across, close to the noise floor.
- Low input resolution caps OCR reliability regardless of upscaling — an
  upscaled 5 px glyph is still a reconstruction, not new information.
- Hand-drawn/noisy scans are out of scope for the classical CV stages as designed.
- Curved or diagonal wire routing is handled worse than orthogonal routing.
- Implicit reference nodes (e.g. multiple ground symbols) — decide whether
  to merge into one logical node or keep as separate instances (see
  05_open_questions_and_assumptions.md).
- No accuracy metric is reported: two drawings, no annotations. Validation is
  self-consistency plus visual review.
