# The P&ID domain pack — genericity, demonstrated

The brief ships five drawings. D4 and D5 are electronic schematics; **D1,
D2 and D3 are process & instrumentation diagrams** — vessels, centrifugal
pumps, control valves, instrument bubbles, heat exchangers — a different
symbol vocabulary and different line conventions. The original pipeline
answered two of the five.

The rubric rewards "a generic, scalable approach" over raw accuracy. So
rather than tune D4/D5 further, the pipeline was split into **domain-
agnostic machinery** and **per-drawing-type knowledge packs**, and a
second pack was built for P&IDs. If the architecture is really generic,
adding a drawing type should be a data operation, not a code change.

## The seam

`edp/domains/base.py` defines `DomainPack`: a bundle of everything the
stages consult that varies by drawing type —

| pack field | electronic | pid |
|---|---|---|
| `detector_weights` | KiCad-synthetic YOLO (17 cls) | ISA-synthetic YOLO (10 cls) |
| `reference_dir` | KiCad-rendered symbols | procedurally-drawn ISA symbols |
| `designators_path` | R/C/L… (IEC 61346) | PC/LT/FCV… (ISA-5.1) |
| `probe_model_path` | linear probe | — |
| `specialists_module` | battery/cap, BJT/pot | — (empty, documented) |
| `evidence_weights` | yolo 1.2 … | text_prior 1.2 (ISA tags are diagnostic) |

**Not one line of `preprocess.py`, `classify.py`, `ocr.py`, `wires.py` or
`emit.py` changed.** `pipeline.py` gained a `route` stage and passes a
`pack` into `detect_candidates` / `classify_candidates`; `match.py` reads
its fusion weights and specialist groups from the pack. That's the whole
diff to the machinery.

## The P&ID pack

- **Symbols** (`scripts/build_pid_reference.py`): 13 ISA-5.1 symbols
  across 10 classes — Vessel (vertical + horizontal), Pump_Centrifugal,
  Valve_Gate / Valve_Control / Valve_Check, Instrument (field + shared),
  Heat_Exchanger (coil + shell-and-tube), Compressor, Filter, Motor —
  drawn directly with OpenCV primitives, each with process-connection
  terminal points. Same rationale as `generate_procedural_variants.py`
  for electronic: P&ID equipment geometry is simple and its connection
  points are far less standardised than KiCad pins, so drawing them beats
  sourcing and rasterising an SVG library.
- **Detector**: `generate_synthetic_dataset.py` (parametrised on
  `--reference-dir` / `--size-min/max`) composites those symbols onto
  blank canvases with wire and text clutter → 900 train / 100 val,
  labels a byproduct of placement. yolov8n, 45 epochs. Zero hand-labeled
  data, same discipline as the electronic detector.
- **Tag prior** (`tags.yaml`): ISA-5.1 convention — first letter =
  measured variable, succeeding letters = function; every `<var><func>`
  form is the same bubble, so they all map to `Instrument`. Control-valve
  tags (`FCV`, `LCV`, …) → `Valve_Control`. Equipment prefixes
  (`P`→pump, `E`/`H`→exchanger, `V`/`D`/`T`→vessel, …).

## The page router

`edp/domains/base.py::route()` picks the pack from the drawing's text
alone. Each pack contributes its designator prefixes plus a
`router_keywords` list (process vocabulary: GPM, OUTLET, STEAM,
CONDENSATE, VESSEL, … vs. electronic: OHM, UF, VDC, BC5, …). The
drawing's OCR tokens are scored against both; higher wins; a thin or tied
signal falls back to a configured default. OCR runs once and the tokens
are handed to the rest of the pipeline.

**Measured on all five drawings:**

| drawing | electronic score | pid score | routed |
|---|---|---|---|
| D1 | 1 | 19 | pid ✓ |
| D2 | 5 | 23 | pid ✓ |
| D3 | 0 | 76 | pid ✓ |
| D4 | 61 | 18 | electronic ✓ |
| D5 | 17 | 3 | electronic ✓ |

`domain: auto` is the default; electronic `edp eval` is unchanged under it
(F1 0.879, classification 65.5%).

## First-pass P&ID results

D1–D3 run end to end and produce the JSON + graph. Qualitatively (see
`outputs/pid_run/*_overlay.png`): instrument bubbles, gate/control
valves, centrifugal pumps, heat exchangers and vessels are detected and
mostly typed correctly on D3; the large horizontal separator on D1 is
missed by the first detector (bigger than its synthetic symbol-size
range — fixed in a v2 retrain with a wider range). Confusions are the
expected ones — bubble vs. exchanger and motor vs. exchanger when both
are circular — exactly what a P&ID geometry specialist would target, and
`edp/domains/pid/specialists.py` is stubbed for that with the specific
pairs named.

A hand-verified `data/golden/D1.json` (or D3) and a per-domain
`edp eval` number is the honest next measurement — built the same way as
D4/D5 (id-only overlay, cross-referenced against the drawing).

## Why this is the answer to "generic and scalable"

The claim isn't "it's accurate on P&IDs" — it's that a genuinely
different drawing type was added as **a folder of data plus one 150-line
symbol script**, with the eight-stage pipeline, the evidence-fusion
classifier, the connectivity graph and the output contract all untouched
and still passing their tests. That is the property the brief asks for,
shown rather than asserted.
