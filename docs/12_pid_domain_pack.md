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

## First-pass P&ID results — measured

`data/golden/D3.json` is a hand-verified golden set (21 real components,
built the same way as D4/D5: id-only overlay cross-referenced against the
drawing). `edp eval` on it, with the synthetic P&ID detector and the pid
pack, **no machinery changes**:

| | D3 |
|---|---|
| Detection precision | 0.61 |
| Detection recall | **0.95** (20 / 21) |
| Detection F1 | 0.74 |
| Classification (matched pairs) | 0.55 |

**Recall 0.95 is the headline** — a detector trained only on 900
procedurally-drawn ISA symbols found almost every real component in a P&ID
it never saw, which is the "does the generic approach transfer" question
answered directly.

The two gaps are specific and diagnosable, not vague:

1. **Precision (13 false positives)** — boxes on descriptive text
   ("Centrifugal Pump", "Air Cooled Exchanger"), on flow arrows, and a few
   duplicates. The electronic path suppresses text regions before
   localization (`_strip_text`) and dedups (`merge_overlapping`); wiring a
   generic OCR-token / arrow suppression into the YOLO path and tightening
   the merge would remove most of these. Machinery change, so measured and
   gated like any other.
2. **Classification (0.55)** — the confusions are almost entirely
   *circular-equipment*: instrument bubbles (FCV/LCV/HCV) → Heat_Exchanger
   (3 of 9 errors), Accumulator → Instrument, Motor → Compressor (2). A
   `bubble-vs-exchanger` specialist (internal zigzag/coil vs. a horizontal
   divider line or a bare letter-tag) and a `motor` check ("M" glyph
   present) are exactly the hand-coded-convention case the electronic
   specialists already established — `edp/domains/pid/specialists.py` is
   stubbed with these pairs named.

One false negative: the Tray Column (detected, but the box drifted below
IoU 0.5 against the hand-drawn extent).

## Why this is the answer to "generic and scalable"

The claim isn't "it's accurate on P&IDs" — it's that a genuinely
different drawing type was added as **a folder of data plus one 150-line
symbol script**, with the eight-stage pipeline, the evidence-fusion
classifier, the connectivity graph and the output contract all untouched
and still passing their tests. That is the property the brief asks for,
shown rather than asserted.
