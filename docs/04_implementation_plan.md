# Implementation Plan (2–3 Day Timeframe)

Structure, config and CLI contract are specified in `07_project_layout.md`;
the intermediate objects are specified in `06_data_model.md`. This doc is
the schedule only.

## Sequencing principle
Build the **thin end-to-end slice first**, then deepen each stage. A pipeline
that produces a bad JSON and a bad graph on day 1 is far more valuable than
one with an excellent localizer and no output path on day 3 — it makes every
later change measurable, and it guarantees a deliverable exists.

## Day 1 — Skeleton pipeline, end to end

**Morning: walking skeleton**
- [ ] Repo scaffold per `07_project_layout.md`; pin deps (OpenCV, scikit-image,
      PyTorch + DINOv2, pytesseract, NetworkX, pydantic)
- [ ] `types.py`: Symbol, Terminal, Net, TextToken dataclasses
- [ ] Stub every stage to pass typed objects straight through
- [ ] `edp run data/raw/D5.png` emits a valid (empty) JSON + empty graph
- [ ] Debug-overlay renderer (`--stage`) working early — every later stage is
      tuned by looking at its overlay

**Afternoon: localization + classification**
- [ ] Preprocessing: grayscale, adaptive binarize, deskew, colour-layer split
- [ ] Candidate localization: connected components/contours; morphological
      closing for the dashed SHIELD box in D5
- [ ] Reference library: crop exemplars from D4/D5, **rotation-augment**
      (0/90/180/270 + mirror), embed with DINOv2, store index
- [ ] Terminal templates for the high-value classes (BJT, IC, transformer,
      MOSFET) in normalised crop coordinates
- [ ] Classification: embed candidates, NN match, unknown threshold
- [ ] Checkpoint — D5: are R1–R6, C1–C8, Q1, XTAL, T1, L2–L5, grounds found?
      D4: are the circled BJTs matched by the same library as D5 bare Q1?
      (That cross-drawing match is the generalization claim in miniature.)

## Day 2 — Text, wires, connectivity

**Morning: OCR**
- [ ] 3–4× Lanczos upscale, then Tesseract; multi-orientation passes (0/90/270)
- [ ] Token → symbol proximity association
- [ ] Token splitting: "R1 10K" → id + value, "XTAL149.89MHz" → type + value
- [ ] Checkpoint — are R/C/L designators and values on D5 mostly right?

**Afternoon: wires and nets** (highest-risk block; keep it uninterrupted)
- [ ] Symbol subtraction, skeletonization, skeleton-graph construction
- [ ] Junction-dot detection (filled-circle match, radius 2–5 px)
- [ ] Degree-4 crossing decomposition — split undotted crossings into two
      pass-through paths **before** taking connected components
- [ ] Connected components → Net objects; snap terminals within radius
- [ ] Synthetic unit tests here, not on the real drawings: hand-built images
      of a dotted T-junction and an undotted X-crossing, asserting exact net
      counts. This is the only place the crossing rule can be verified
      precisely, because it is the only place we control the input.
- [ ] Checkpoint — D5: does the top rail form one net over C1/R1/IC without
      absorbing unrelated crossings?

## Day 3 — Assembly, validation, documentation

- [ ] JSON generation per `03_json_schema_spec.md` (nets expanded to pairwise
      `connections`; nets + validation blocks emitted alongside)
- [ ] Validation: unattached terminals, <2-terminal nets, isolated symbols,
      duplicate/asymmetric connections, confidence flags
- [ ] Graph: bipartite build, projection to component graph, GraphML +
      node-link JSON + rendered PNG
- [ ] Snapshot the D4/D5 outputs as regression fixtures
- [ ] Full run over both drawings; confidence overlay on the source image
- [ ] Finalise technical documentation: architecture, rationale, trade-offs,
      limitations, future improvements
- [ ] Demo walkthrough — including a **live "add a new symbol class in three
      steps"** demonstration, which is the most direct evidence for the
      scalability criterion

## Stretch (if time remains)
- [ ] VLM fallback for low-confidence junction clusters (targeted crop +
      question, not a pipeline-wide dependency)
- [ ] Ground-node merging as a configurable post-processing pass
- [ ] Netlist export (SPICE-style) as a second projection of the same graph

## Risk register
| risk | likelihood | mitigation |
|---|---|---|
| junction dots ~3–5 px, near noise floor at D5 resolution | high | tune on upscaled image; treat ambiguous crossings as unconnected and flag, rather than guessing |
| OCR fails on smallest labels | high | upscale; fall back to auto-generated ids with `metadata.source` marking the fallback |
| touching symbols merge into one component | medium | area/aspect filters, then split by local minima; document as known limitation |
| DINOv2 download/runtime friction on CPU | medium | cache weights day 1; batch embeddings; ViT-S/14 is sufficient at this scale |
| terminal templates take longer than budgeted | medium | degrade to bbox-edge inferred terminals; pin names are lost but connectivity survives |
