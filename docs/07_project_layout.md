# Project Layout, Configuration, and Extension Path

Scalability and maintainability are explicit scoring criteria, so the
repo structure is a deliverable in its own right, not an afterthought.

## Directory structure

```
electrical-drawing-pipeline/
├── src/edp/
│   ├── config.py             # typed config load/validate (pydantic)
│   ├── types.py              # Symbol, Terminal, Net, TextToken dataclasses
│   ├── pipeline.py           # stage orchestration; the only place order lives
│   ├── preprocess/
│   │   ├── binarize.py       # grayscale, adaptive threshold, denoise
│   │   ├── deskew.py
│   │   └── layers.py         # colour-channel split (D5 blue/black), border strip
│   ├── localize/
│   │   ├── proposals.py      # skeleton branch/endpoint density candidate proposer
│   │   └── morphology.py     # dashed-boundary closing (kind-hint only, see docs/01)
│   ├── detect/                # [experimental branch] YOLO localizer
│   │   └── yolo_detect.py    # drop-in replacement for localize/proposals.py;
│   │                         # falls back to it automatically if no trained
│   │                         # weights exist — see docs/02
│   ├── classify/
│   │   ├── embedder.py       # DINOv2 wrapper (768-dim base model), batched
│   │   ├── faiss_index.py    # [experimental] FAISS IndexFlatIP over the library
│   │   ├── kicad_import.py   # .kicad_sym parser + rasterizer, terminal extraction
│   │   ├── library.py        # reference library build/load, rotation augmentation
│   │   └── match.py          # nearest-neighbour (via FAISS) + unknown thresholding
│   ├── text/
│   │   ├── ocr.py            # upscaling, multi-orientation passes
│   │   └── associate.py      # token → symbol proximity assignment
│   ├── wires/
│   │   ├── skeleton.py       # thinning, skeleton graph construction
│   │   ├── junctions.py      # dot detection, degree-4 crossing decomposition
│   │   └── nets.py           # connected components → Net objects
│   ├── validate/
│   │   └── checks.py         # dangling, isolated, duplicate, confidence rules
│   ├── emit/
│   │   ├── json_out.py       # schema-conformant JSON
│   │   └── graph_out.py      # NetworkX build, GraphML/JSON/PNG export
│   ├── web/
│   │   ├── server.py         # FastAPI: static host + POST /api/process
│   │   └── static/
│   │       └── index.html    # single file, no framework, no build step
│   └── cli.py
├── data/
│   ├── raw/                  # D4.png, D5.png
│   ├── kicad_raw/            # fetched .kicad_sym source files
│   ├── reference/            # rendered symbol PNGs + terminal templates,
│   │                         # one dir per class — output of build_reference_from_kicad.py
│   ├── synth/                # [experimental, gitignored] generated YOLO training set —
│   │                          # output of generate_synthetic_dataset.py, fully regeneratable
│   ├── synth_dense/          # [experimental, gitignored] denser-packing training variant
│   ├── synth_holdout/        # [experimental, gitignored] held-out synthetic sanity check —
│   │                          # NOT a substitute for real validation (see docs/02): same
│   │                          # generation distribution as training, so it only confirms
│   │                          # in-distribution generalization, not real-world transfer
│   ├── validation/            # REAL circuit diagrams for actual validation: D4, D5, plus
│   │                           # 50 downloaded sample circuits in a different drawing
│   │                           # convention than KiCad's. No ground-truth boxes exist for
│   │                           # these; evaluated by visual audit (validate_on_real.py).
│   │                           # Tests topology-realism AND icon-style transfer together —
│   │                           # a failure here doesn't say which one is at fault.
│   └── validation_kicad_topology/  # [experimental, gitignored] synthetic but topologically
│                                    # realistic circuits (ladder network: rails + vertical
│                                    # branches, junction dots — see generate_ladder_circuits.py)
│                                    # built from our own KiCad-rendered symbols. Isolates the
│                                    # topology question from the icon-style question: same
│                                    # style as training, so a failure here can only be a
│                                    # topology/context problem, not a style-mismatch one. Has
│                                    # real YOLO ground-truth labels (we placed everything), so
│                                    # this one CAN be scored with automated mAP, unlike
│                                    # data/validation/'s real circuits.
├── scripts/
│   ├── build_reference_from_kicad.py   # .kicad_sym -> data/reference/ (see docs/05)
│   ├── generate_synthetic_dataset.py   # [experimental] composites reference symbols at
│   │                                   # random positions into labeled YOLO training scenes
│   ├── generate_ladder_circuits.py     # [experimental] composites reference symbols into
│   │                                   # topologically realistic ladder-network circuits —
│   │                                   # see data/validation_kicad_topology/ above
│   ├── train_yolo.py                   # [experimental] trains the YOLO localizer
│   └── validate_on_real.py             # [experimental] runs the full pipeline over
│                                        # data/validation/, writes overlays for audit
├── config/
│   └── default.yaml
├── outputs/                  # per-drawing JSON, graphs, debug overlays;
│                              # yolo_runs/ holds training checkpoints [experimental]
├── tests/
├── notebooks/                # exploration only, never imported by src
└── docs/
```

**The rule that keeps it maintainable:** each stage module takes typed
objects in and returns typed objects out, with no knowledge of what runs
before or after it. `pipeline.py` is the only file that knows the stage
order. That is what makes a stage independently testable, independently
swappable (e.g. Tesseract → PaddleOCR), and independently debuggable.

## Configuration

Every threshold lives in `config/default.yaml`, never inline in code —
tuning is the bulk of the work on this kind of pipeline, and buried magic
numbers make it unrepeatable.

```yaml
preprocess:
  binarize_block_size: 31
  deskew_max_angle_deg: 5
localize:
  min_component_area: 80
  max_component_area: 40000
  dash_close_kernel: 5
classify:
  model: dinov2_vits14
  rotations: [0, 90, 180, 270]
  unknown_similarity_threshold: 0.62
text:
  upscale_factor: 3
  orientations: [0, 90, 270]
  max_association_distance: 45
wires:
  junction_dot_min_radius: 2
  junction_dot_max_radius: 5
  terminal_snap_radius: 12
```

## CLI contract

```bash
edp run data/raw/D5.png --config config/default.yaml --out outputs/
edp run data/raw/ --out outputs/              # batch over a directory
edp build-library data/reference/ --out data/reference/index.npz
edp debug data/raw/D5.png --stage wires       # overlay render for one stage
edp serve --port 8000                         # demo frontend
```

`--stage` debug rendering is not a nicety: on a pipeline with this many
visual stages, being able to see the skeleton or the proposals overlaid on
the source is the difference between tuning in minutes and tuning blind.

## Demo frontend

A single static `index.html` plus a thin FastAPI wrapper around the same
`pipeline.run()` the CLI calls. Its only job is to make the pipeline
demonstrable in the review without a terminal.

```
┌─────────────────────────────────────────────────┐
│  [ choose file ]  D5.png          [ Process ]   │
├────────────────────────┬────────────────────────┤
│  JSON                  │  GRAPH                 │
│  ┌──────────────────┐  │  ┌──────────────────┐  │
│  │ {                │  │  │                  │  │
│  │  "symbols": [    │  │  │   (rendered      │  │
│  │   { "id": "R1",  │  │  │    component     │  │
│  │     "type": ...  │  │  │    graph PNG)    │  │
│  │  ...             │  │  │                  │  │
│  └──────────────────┘  │  └──────────────────┘  │
└────────────────────────┴────────────────────────┘
```

**Contract:** `POST /api/process` takes the uploaded image, returns
```json
{"json": { ...schema-conformant output... },
 "graph_png": "data:image/png;base64,...",
 "overlay_png": "data:image/png;base64,...",
 "timing": {"localize": 0.4, "classify": 2.1, "wires": 0.9}}
```

**Deliberate constraints:**
- **No framework, no build step.** One HTML file with inline CSS and vanilla
  JS, `<pre>` for the JSON and `<img>` for the graph. It is a demo surface,
  not a product; a toolchain here would be cost with no evaluation credit.
- **Graph rendered server-side** as PNG, reusing the existing `graph.png`
  export from `emit/graph_out.py`. A client-side renderer (cytoscape, vis.js)
  would mean a second graph implementation to keep in sync with the first.
- **The web layer holds no pipeline logic.** `server.py` calls the same
  entry point as `cli.py` and formats the result. If the two could drift,
  the demo stops being evidence that the pipeline works.
- `overlay_png` (source image with bboxes and confidence) is the most
  useful thing on screen during a live walkthrough — it shows *why* the
  JSON says what it says. Worth building even though it was not requested.

## Adding a new symbol type (the scalability claim, concretely)

1. Drop 1–3 crops into `data/reference/<new_class>/`
2. Run `edp build-library`
3. Done — no code change, no retraining.

This three-step path is the concrete form of the "generic and scalable"
argument in `02_model_selection_rationale.md`. It is worth demonstrating
live in the presentation.

## Testing strategy

Given there is no labelled ground truth (two drawings, no annotations),
tests target **mechanism, not accuracy**:

- **Unit, synthetic**: generate small images with known content — a
  T-junction with a dot, an X-crossing without one, two boxes and a wire —
  and assert the net decomposition produces exactly the expected nets.
  This is where the crossing rule gets verified precisely, because we
  control the input.
- **Contract**: emitted JSON validates against the schema; every id in
  `connections` exists in `symbols`; connections are symmetric.
- **Regression**: pin the current D4/D5 output as a snapshot so tuning
  changes surface as an explicit diff rather than a silent drift.
- **Not attempted**: a precision/recall number. With two unlabelled
  drawings any such figure would be manually-scored anecdote, and
  presenting it as a metric would be dishonest. Stated as a limitation.
