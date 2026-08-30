# Electrical Drawing Interpretation Pipeline — Technical Documentation

Pratik Shrivastava

---

## 1. What this is

The task was to take a raster image of an electrical drawing and read it the
way an engineer would: find every symbol, work out what each one is, figure
out how they're wired together, and hand back a structured netlist plus a
graph. Five drawings were provided. Two of them (D4, D5) are electronic
circuit schematics — resistors, transistors, a MOSFET, an optocoupler. The
other three (D1, D2, D3) are process & instrumentation diagrams: vessels,
centrifugal pumps, control valves, instrument bubbles, heat exchangers. Same
brief, genuinely different symbol vocabularies and drawing conventions.

The first version of this pipeline only handled the two schematics, because
that's what I started looking at. The bulk of the interesting engineering
happened when I went back and made it handle both kinds of drawing without
rewriting the pipeline — and that restructuring is what most of this document
is about, because it's the part I'd actually defend in a review.

The brief is explicit that raw accuracy isn't the main thing being scored —
it's whether the approach is generic and scalable, and whether every decision
along the way can be justified. I took that seriously. A lot of what follows
is me explaining why I *didn't* do the thing that would have bumped a number.

**Running it:** `conda env create -f environment.yml`, `pip install -e .`,
`bash scripts/bootstrap.sh`, then `edp run data/validation/D4.png --out
outputs/` or `edp serve`. The shipped model weights and all symbol assets
are in the repo; retraining from scratch is documented in `README.md`. The
design history and every measured experiment referenced below are in
`docs/08`–`docs/13`.

---

## 2. The core idea: separate the machinery from the knowledge

Here's the problem I kept running into. A pipeline stage like "binarize the
image" or "skeletonize the wires" or "fuse the classifier votes" is identical
whether you're looking at a circuit or a P&ID. But "what does a resistor look
like", "what does the letter R mean on a label", "is a dot on a crossing a
connection" — those are drawing-type-specific, and in the first version they
were scattered through the stage code as hardcoded paths, magic strings and
if-branches. So the only way to support a P&ID was to go edit the machinery,
which felt wrong, because the machinery wasn't the part that needed to change.

The fix was to pull all the drawing-type-specific knowledge into a thing I
call a **domain pack** — a folder plus a small config object. Everything the
stages consult that varies by drawing type lives in the pack:

| what varies | electronic pack | P&ID pack |
|---|---|---|
| which trained detector to run | KiCad-synthetic YOLO, 17 classes | ISA-synthetic YOLO, 10 classes |
| the reference symbol library | KiCad-rendered symbols | procedurally-drawn ISA symbols |
| the OCR text prior | R, C, ZD… (IEC 61346) | PC, LT, FCV… (ISA-5.1) |
| the linear-probe classifier | trained artifact | none |
| geometry specialists | battery/capacitor, BJT/potentiometer | none yet (stubbed, pairs named) |
| evidence-fusion weights | YOLO gets a modest edge | OCR gets the edge (ISA tags are diagnostic) |

The machinery — preprocessing, localization, the evidence-fusion classifier,
OCR association, the connectivity graph, the JSON and graph output, the
evaluation harness — is completely domain-agnostic. When I added the P&ID
pack, not one line of `preprocess.py`, `classify/match.py`, `ocr.py`,
`wires.py` or `emit.py` changed. The pipeline orchestrator gained one new
stage (routing, below) and started passing a `pack` object into two function
calls. That was the entire diff to the machinery.

I think of it like a spell-checker. The engine that finds word boundaries,
compares against a list, underlines the misses and ranks suggestions is the
same for every language. The word list and grammar rules are a per-language
pack. Adding Spanish support doesn't mean rewriting the spell-checker. That's
the property the brief is asking for when it says "generic and scalable", and
building the P&ID pack was me proving the pipeline actually has it, rather
than just claiming it.

Concretely, a pack is `edp/domains/<name>/pack.yaml` — a few paths and
weights — plus, next to it, the small knowledge files that pack references (a
`specialists.py`, a `designators.yaml` or `tags.yaml`). Large regeneratable
assets — the reference image libraries, the trained weights — stay under
`data/` and are pointed at by path. `edp/domains/base.py` loads a pack and
exposes it as a frozen dataclass. Adding a third drawing type later — a
single-line power diagram, say — is a new folder. No core code.

---

## 3. Walking through the pipeline

An image comes in and goes through eight stages. `edp/pipeline.py` is the
only file that knows the full order and why it's that order; every stage
module takes typed objects in and returns typed objects out, with no
knowledge of what ran before or after it. That isolation is deliberate — when
something goes wrong I want exactly one place it could have gone wrong.

### Preprocess

Grayscale, then an adaptive (local-mean) threshold rather than one global
cutoff, because contrast isn't always even across a page and I didn't want a
threshold I'd have to retune per drawing. Then a small-angle deskew (these
are digitally-generated drawings, not scanned pages, so I'm correcting a few
degrees of export tilt, not building rotation invariance). Then speckle
removal — and this one bit me. My first instinct was a morphological opening.
I rendered the binarized output and watched it erase almost every
one-pixel-wide wire and symbol line in the drawing, because a line that thin
can't contain even a tiny structuring element anywhere along its length. The
junction dots survived; the actual circuit didn't. Switched to
connected-component area filtering, which judges a blob by its size, not its
width, so a long thin wire and a fleck of noise get treated differently.

One drawing (D5) draws some conductors in blue. There's an optional colour
mask that pulls blue ink out as a separate signal. It's explicitly
opportunistic — on a monochrome drawing it returns empty, and everything
downstream is written to treat "no colour signal" as normal, not an error,
because half the input has no colour.

### Route

New stage. If the domain is set to `auto` (the default), the pipeline decides
whether this is a circuit or a P&ID from the drawing's text alone. Each pack
contributes its designator prefixes plus a keyword list — circuits have OHM,
UF, VDC, part numbers like BC547; P&IDs have GPM, OUTLET, CONDENSATE, VESSEL,
and ISA instrument tags. The OCR tokens are scored against both packs, higher
wins, and a thin or tied signal falls back to a configured default so a
text-sparse drawing still runs. OCR runs once here and the tokens are handed
to the rest of the pipeline so it isn't run twice.

I tried this on all five drawings. D1/D2/D3 route to P&ID with scores of
19/23/76 against 1/5/0 for electronic; D4/D5 route to electronic 61/17
against 18/3. Wide margins, no ambiguity. It's a cheap, explainable rule —
no model, no training — and it works because the two kinds of drawing
genuinely say different things on them.

### Localize

The shipped path is a YOLOv8n detector, and I generated all of its training
data myself. There's a classical density-based localizer as a fallback if no
trained weights are present, so the pipeline stays runnable before any model
exists — that "walking skeleton" discipline is threaded through the whole
codebase (empty reference library, missing probe artifact, missing detector
all degrade rather than crash).

The classical localizer was the original approach — connected-component
analysis over skeleton branch/endpoint density. I shipped it early because it
needs no training data. It fell apart when I looked at the output: roughly
six out of ten boxes were wrong, mostly wire bends and rail junctions that
looked "symbol-shaped enough" to a density heuristic, and it missed real
components too. I could have kept tuning thresholds but I'd have been chasing
noise. So I trained a proper detector — YOLO, because it learns actual symbol
shapes and the nano variant is small enough to retrain in minutes on a laptop
GPU whenever the reference library changes, which mattered given how many
iterations this went through.

How the training data is made: KiCad-rendered reference symbols get pasted
onto blank canvases at the real drawings' size distribution (measured from
D4/D5, not guessed), rotated 0/90/180/270, with synthetic Manhattan-routed
wire clutter and dash-cluster text clutter added as *unlabelled* negatives —
so the detector learns not to fire on wires and text without ever being told
"this is not a symbol", it just never sees a label there. The bounding-box
label is a byproduct of where the generator put each symbol. Nobody drew a
box. The shipped model mixes three generators: sparse scatter, dense packing,
and a topology-realistic "ladder" generator with rails and branches and
junction dots, built as a separate third generator specifically so a low
score would be attributable — "doesn't handle real topology" versus "doesn't
generalize to unfamiliar icon styles" are different problems and one harder
scatter generator would have conflated them.

The P&ID detector is the same story with a different symbol source (§8).

### Classify

This is the part I spent the most time on and ended up furthest from my first
instinct. Details in §4.2. The short version: every candidate box gets voted
on by up to five independent sources, weighted by each source's own
confidence in *this particular reading*, and any source is allowed to abstain
rather than guess.

### Text association

Tesseract, run across three page-segmentation modes and three orientations
per crop, keeping every reading without deduplicating. Details in §5. The
output resolves instance identity (which resistor is R1 versus R2) and reads
component values.

### Wires

Subtract the classified symbol *ink* from the binary image (keeping wires
that merely pass through a box), skeletonize what's left to one-pixel-wide
lines, re-place each symbol's terminals onto the point where a wire actually
touches its bounding box, build a pixel graph, and walk it into nets. The
one rule that does real work here: a wire crossing is only a connection if
there's a dot drawn on it. Details in §6.

### Validate

Self-consistency checks — symbols with terminals but no net, terminals that
never snapped to anything, nets with only one member. Not an accuracy score
(there's no ground truth for an arbitrary new drawing); it's there to catch
structural nonsense before it reaches the output.

### Emit

The JSON matches the schema in the brief exactly — id, type, coordinates,
connections, nothing decorative. The graph is built bipartite internally
(symbol nodes and net nodes as distinct types) because a five-terminal net
expanded straight into a symbol-to-symbol graph becomes a ten-edge clique
implying wires that don't exist. The delivered component graph is projected
from the same trimmed JSON the file output uses, not recomputed
independently, so the picture and the data can't silently disagree.

---

## 4. Model selection rationale

### 4.1 Localization — YOLOv8n

I went with YOLOv8n over the classical-only approach because the classical
one had a measured ~60% false-positive rate on D4 and no notion of what a
resistor should look like versus noise. YOLO gave me that, and the nano
variant (about 3M parameters) trains in minutes on a 4GB laptop GPU. The
trade-off I accepted going in, and which held for the whole project, is that
it's only as good as its synthetic training data — it never saw a real
drawing until evaluation, so its recall on unusual real-world symbol styles
is the weakest link in the pipeline. Two attempts to fix that with more or
different training data both regressed and were reverted (§10).

I did *not* use YOLO's own class predictions as final. It's trained purely on
synthetic composites, and fine-grained class distinctions — NPN versus PNP,
diode versus zener — are exactly where the synthetic-to-real gap bites
hardest. So YOLO's class and confidence go in as one vote among five, not
the answer.

### 4.2 Classification — five-source evidence fusion

I started with one method: DINOv2 embeddings matched against a reference
library by nearest neighbour. No labelled training data, just example crops
of each symbol type. It worked, but it made confident, silent mistakes on
visually similar symbols — capacitor versus battery, BJT versus
potentiometer — and I had no way to know when it was guessing versus sure.

My next instinct was to let YOLO only localize and have DINOv2 name
everything, since I didn't fully trust YOLO's synthetic-trained class head.
But when I compared the two side by side on D4, they were wrong on completely
different symbols. YOLO got about 10 of 16 right, DINOv2 about 7 of 16, and
the errors barely overlapped. Discarding either one was throwing away real
information. So classification stopped being one model's call and became a
weighted vote.

Each source produces a `ClassificationEvidence`: a dict of `{class: score}`,
a scalar confidence in *this specific reading*, and some metadata for
explainability. Or it produces an explicit "no evidence" — abstaining, not
guessing. Fusion sums, per class:

    total[class] += weight[source] × evidence.confidence × evidence.class_scores[class]

then takes the argmax. Two things this buys over a fixed priority order like
"OCR beats DINOv2 beats YOLO":

- A source's own certainty about this reading matters, not just a fixed
  per-source weight. A garbled OCR read contributes almost nothing even
  though OCR as a source has real weight.
- Two sources agreeing pushes a class up additively. Agreement is rewarded
  structurally, not with a special-cased rule.

The five sources:

1. **YOLO's class head** — `{yolo_class: 1.0}` at the detector's own
   confidence. Never abstains, but a low-confidence box contributes almost
   nothing.

2. **DINOv2 nearest-neighbour** — each candidate crop is embedded and matched
   by cosine similarity against a library of embedded reference symbols.
   Returns the top-3 *distinct classes*, over-fetching 20 raw neighbours
   first so many rotated copies of one class don't crowd out a genuinely
   different second candidate. Below a similarity threshold it abstains
   entirely — a weak match is worse than no match, because it would vote
   confidently for a wrong class.

   Why DINOv2 and not CLIP: DINOv2 is trained for dense visual similarity,
   which is better for near-duplicate shape matching of technical line art.
   CLIP is trained for image-text alignment and is measurably weaker at
   telling a resistor's zigzag from an inductor's coil, because in its
   image-text space those are "similar enough".

   Why nearest-neighbour and not a fine-tuned classifier: this is the single
   decision that makes the system generic in the sense the brief asks for.
   Adding a symbol class is dropping in reference crops and rebuilding an
   index — no retraining, no new labelled dataset, no GPU time. A fine-tuned
   classifier would need retraining (with the regression risk demonstrated
   twice in §10) every time the class set changes.

   DINOv2 embeddings aren't rotation-invariant, so the library is augmented
   at build time with four rotations and a mirror, and the matched rotation
   is *kept* — it's what orients the terminal template onto the candidate box
   later.

3. **Linear probe** — a scikit-learn logistic regression trained on frozen
   DINOv2 embeddings of domain-randomised synthetic crops. A learned decision
   boundary over the same embedding space the nearest-neighbour match uses,
   built as an independent hypothesis: does a learned boundary generalize
   better than "closest single reference point"? Kept as a genuinely separate
   vote rather than a replacement — if it's worse on some case it can only
   add a wrong vote that a correct nearest-neighbour vote outweighs, never
   override a good source. Reuses the embedding already computed for source 2,
   so it's nearly free.

4. **OCR reference-designator / part-number prior** — engineering designators
   are a documented convention (IEC 61346, ANSI Y32.2): R for resistor, C for
   capacitor, ZD for zener. Specific part numbers are stronger still — BC547
   is unambiguously an NPN transistor. This is knowledge no vision model has.
   Three confidence tiers: exact part number is near-authoritative (0.95), a
   clean designator is a moderate vote (0.55), anything unparseable
   contributes nothing at all. That last part is a deliberate rule — never
   let garbled OCR outvote confident visual evidence by treating noise as
   weak-but-nonzero. The table is a YAML file, not code; adding a convention
   is a data edit. Some designators are deliberately absent — T means
   Transformer in one drawing and Transistor in another, so a prior there
   would be wrong half the time by the convention it's meant to encode.

   For the P&ID pack this same mechanism carries the ISA-5.1 tag convention:
   first letter is the measured variable (P, L, F, T), succeeding letters are
   the function (I, C, T, V), every such tag is drawn as the same bubble so
   they all map to the Instrument class.

5. **Geometry specialists** — hand-coded checks that only run when fusion is
   already ambiguous between classes in a known confusion group. Battery
   versus capacitor is decided by plate count and curvature; BJT versus
   potentiometer by whether there's a circle. Hand-coding is right here
   specifically because the distinguishing feature is a *documented drawing
   convention*, not a learned statistical pattern.

   One specialist — NPN versus PNP by arrowhead direction — is built,
   unit-tested, and deliberately *not* wired in. It's 2/2 correct on clean
   reference renders but 1/3 on real D4 crops, worse than chance. I left it
   in the module, documented, and excluded from routing rather than shipping
   it anyway. That's one of the stronger "why should I trust your judgment"
   answers I have — the same measure-before-trusting discipline applied even
   to work that was fully built and worked in principle.

The default fusion weights (YOLO 1.2, everything else 1.0) are a single
reasoned default, not a grid search. The reasoning — YOLO is the supervised,
in-domain signal so it gets a modest, not dominant, edge — is real and was
checked for not regressing anything. The specific value 1.2 wasn't swept.
That's a genuine gap; the fusion *architecture* is the justified decision,
the exact weight is a reasonable default that hasn't hurt.

### 4.3 OCR — Tesseract, multi-pass

Deterministic, dependency-light, adequate on clean synthetic-quality glyphs
once upscaled. I upscale 3× with Lanczos first, because D5's labels are only
a few pixels tall natively and Tesseract has an effective floor around 20px
of glyph height. That's the single highest-impact OCR decision. Honest
caveat: upscaling reconstructs, it doesn't add information — a genuinely
sub-5px glyph stays unreliable.

My first version ran Tesseract once per region and missed anything rotated or
anything its default mode wasn't tuned for. Now it runs three PSM
configurations across three orientations and keeps *every* reading without
deduplicating. I measured every PSM × upscale combination against known
designators on both drawings first — no single config won on both, and
merging all three beat every individual config. I tried deduplicating by
keeping the highest-confidence reading per region and it *lost* a real answer
(a correct "$1"→"S1" switch designator crowded out by a higher-confidence but
wrong reading from a different pass) — Tesseract's confidence isn't a
reliable "which reading is right" signal for short garbled technical tokens.
So I pass everything through and let downstream nearest-token matching surface
the correct one redundantly.

### 4.4 Connectivity — classical CV, not a learned model

These drawings are thin, high-contrast, mostly-orthogonal strokes — the
textbook case for skeletonization. No training data exists for a learned
segmentation model and none is needed for this input. A classical approach
stays fully deterministic and traceable to a specific pixel rule, which
matters for the same "justify every decision" reason as everything else. I
ruled out detecting wires as another YOLO class early — wires aren't compact
objects, they're long thin structures of arbitrary shape, and a bounding box
is the wrong representation.

### 4.5 Where a VLM or LLM was, and wasn't, used

Not in the pipeline, and that was deliberate. Early on I looked at prompting
a vision-language model to read the whole drawing and output structured
symbols and connections directly. It's tempting because it needs almost no
engineering. But I couldn't get it to a place where I trusted the output —
no clear notion of confidence, no way to know *why* it made a call, no way to
systematically improve it when it got something wrong beyond rewriting the
prompt and hoping. The brief also explicitly lists preprocessing logic,
detection strategy, connectivity inference, post-processing and validation,
JSON generation and graph construction as things I must develop — a VLM call
that returns a finished description doesn't produce any of those as
inspectable stages, and all six would still need to exist and be justified
around it.

Where I did use an LLM: authoring the reference-designator and ISA-tag YAML
tables once, at build time, as a documentation aid. The shipped system reads
a static file on every request, never a model. If asked "did you use an LLM"
the honest answer is yes, as an authoring aid, never as a runtime dependency.

The place I'd actually bring an LLM in, if I extended this, is post-hoc — a
sanity-check layer reviewing the final JSON for an implausible component
value or a mislabelled net, not the core extraction. That's a real scoped
next step, not a rejected idea; it's just not warranted until there's
evidence the deterministic methods have plateaued.

---

## 5. Pre-processing and post-processing

Preprocessing is covered in §3. The one thing worth repeating is the
denoise-by-component-area decision, because a morphological open — the
obvious choice — actively destroys these drawings, and I only caught it by
rendering the intermediate output and looking.

On the post-processing side:

**Duplicate merging.** Both localizers occasionally produce two or three
boxes for one physical symbol — the density one fragments, and YOLO's own NMS
doesn't catch every near-duplicate from different anchor scales. I merge any
pair of boxes whose overlap passes a threshold, but I don't blindly discard
the loser — I keep whichever duplicate had the higher classification
confidence, because I found a real case where two duplicate boxes for the
same transistor scored two different classes and the more confident one was
correct. When the real-data detector (§10) is enabled, this same merge is
what folds its class-agnostic boxes in alongside YOLO's class-bearing ones.

**Consistency checks.** After the pipeline runs, I check the result against
itself — is every symbol with terminals actually on a net, is every terminal
accounted for, is any net down to a single member. None of this is against
ground truth. It's a self-consistency pass to catch structural nonsense that
would embarrass the output even if I never saw the source image.

**Confidence and abstention.** Every symbol carries the fused confidence plus
a margin (top guess minus runner-up, on a normalised scale). A low margin is
the system saying "I chose this but it was close". Symbols under a minimum
confidence get flagged in the validation report rather than silently passed
through as certain. I don't auto-discard them — a visible low-confidence
symbol is more useful to a reviewer than a hidden gap or a falsely confident
wrong answer.

---

## 6. Connectivity extraction

This is the stage where getting a subtle rule wrong doesn't crash anything,
it just quietly produces a wrong but plausible answer. So it's the one I was
most paranoid about.

**Wire detection.** Clear each classified symbol's bounding box from the
binary, skeletonize what's left to one-pixel lines, build a pixel-level
graph — each foreground pixel a node connected to its eight neighbours.
Clearing the whole box used to erase any wire routed near or behind a wide
symbol (D4's MOSFET box is 148×180 px), silently severing nets — so now,
after clearing, any long straight run that enters one side of the box and
leaves the opposite side (a wire passing through, not the symbol's own
body) is restored.

**Junction disambiguation.** I got this wrong first. My initial approach
found junction dots by contour shape — look for small round blobs. It quietly
failed on almost everything, because a real dot sits directly on a wire, so
the dot and the wire become one connected shape and a circle-fit on that
looks nothing like a circle. I counted it: only the handful of dots that
happened to form their own isolated contour were found. What works instead is
local ink density — at every skeleton point where three or more strokes meet,
check how much of a small disk around it is filled. A plain crossing fills
only the width of the two strokes passing through; a drawn dot fills most of
the disk. That's invariant to how many wires run through the point.

Whether a crossing is dotted then does structural work, not just tagging. At
every undotted four-way crossing, before I ever compute connected components,
I split the graph into two independent straight-through paths — the
horizontal pair and the vertical pair, each connected to whichever opposite
arm is closest to 180°. So "two wires crossing without a dot" is disconnected
by construction, not filtered out afterward. A dotted crossing gets a single
synthetic joining node instead.

**Nets, not pairwise links.** A schematic conductor joins N terminals, not
two — one rail touches three or four components. I made the net (an
equivalence class of terminals) the primary object and derive pairwise
`connections` from it at emit time, rather than the reverse, because
reconstructing a net from a pile of pairwise edges is genuinely ambiguous the
moment more than two terminals share a wire.

**Terminals: image-derived, not template-scaled.** The reference entry's
template gives the expected terminal *count* and pin direction, but its
KiCad pin fractions are relative to a render that includes the outward pin
leads — scaled blindly onto the detected box they land inside the body or
past the wire end, which is why the snap radius had crept up to 30 px just
to catch anything. Now a dedicated pass (`wires.refine_terminals`) scans a
thin band just outside each of the four box edges, finds the ink runs
crossing it (a wire connecting to the symbol), and snaps the nearest
template terminal onto each run's midpoint with the edge's outward normal
as its direction. Runs with no matching template terminal become extra
inferred terminals — a 4-pin optocoupler matched to a 2-pin template really
does have 4 connections. With terminals now on the actual wire, the snap
radius was tightened back to 12 px (a sweep from 8 to 30 is now identical).

**The bug that nearly doubled coverage.** KiCad's `.kicad_sym` pin geometry:
`(at x y angle)` is the pin's *outer* electrical connection point, and
`length` walks *inward* toward the body from there. My first importer had
this backwards, so every terminal template pointed at the component body
instead of the wire-contact point. Fixing the two-point swap alone moved
connectivity coverage from roughly half to over 70% on both drawings. I
confirmed the convention against the actual `R_US.kicad_sym` source before
fixing it.

**What this bought, measured.** `edp eval` scores net-level connectivity as
symbol-pair precision/recall against a hand-traced golden
(`data/golden/D4.json`, `_connectivity_verified: true`). On D4, over the
three changes above: **F1 0.40 → 0.64, recall 0.28 → 0.63** — the pipeline
now finds 25 of the 40 hand-traced connections, up from 11. Detection and
classification are untouched. The dominant fix is the image-derived
terminals; the pass-through-wire restoration slightly *hurt* on its own
(the docs-hypothesised fix that measurement didn't back) but recovers 3
more connections once the terminals are right, so it's kept. `docs/13` has
the full table.

**For P&IDs**, the connectivity conventions live in the pack. The obvious
additions — a solid line is a process pipe, a dashed line is an instrument
signal, and an instrument bubble attaches to the valve or line it's drawn
nearest rather than being wired like a component — are scoped but not fully
built. The junction-dot machinery is domain-agnostic and carries over; P&IDs
that use line hops instead of dots would need the pack to say so.

---

## 7. Data — no manually labelled data, anywhere

This is a principle, not a footnote, and it shaped almost every other
decision.

**KiCad as the reference source.** Every electronic symbol image and terminal
template traces back to KiCad's own open `.kicad_sym` libraries (CC-BY-SA),
parsed by a hand-built S-expression parser and rasteriser — not a scraped
icon set, not crops from D4 or D5. Electrical symbols are already a
standardised, zero-licensing-risk ground truth for what each class looks
like.

**Multiple style variants per class.** D4 draws resistors as IEC rectangle
boxes; D5 draws them as ANSI zigzags. Any real corpus disagrees even more.
The library holds both conventions for several classes, each sourced by
fetching the actual alternate KiCad symbol and eyeballing it against real
crops before trusting it. Procedural generation covers the parameters no
fixed symbol expresses — battery cell count, resistor zigzag peak count, a
curved-plate polarised capacitor.

**The synthetic detector training data** is composited from those same
reference symbols, with bounding boxes as a byproduct of placement. Nobody
looked at an image and drew a box.

**The two evaluation drawings are read only for measurement.** D4 and D5 are
read by a human, once, to build the golden ground-truth files, and by the
pipeline to produce predictions to measure. Never by any training script.
This does not violate the no-labelled-data principle — that's about
*training* data, keeping the model generic and retrainable from generated
data. Evaluation has to be grounded in human truth or every number is
unfalsifiable.

**The P&ID symbols** are drawn directly with OpenCV primitives
(`scripts/build_pid_reference.py`), one function per ISA symbol, each with
its process-connection points. Same reasoning as the procedural electronic
variants — P&ID equipment geometry is simple and its connection points are
far less standardised than KiCad pins, so drawing them beats sourcing and
rasterising an SVG library.

**The one exception, and it's disclosed:** the real-data detector experiment
(§10) trained on a public dataset of real, human-labelled circuit drawings.
It's the only place labelled drawing data entered the project, it was run as
an explicit experiment, it measurably hurt, and it's off by default.

---

## 8. The P&ID domain pack, concretely

The pack is a folder: `edp/domains/pid/`.

- **Symbols** — `scripts/build_pid_reference.py` draws 13 ISA-5.1 symbols
  across 10 classes (vessel vertical and horizontal, centrifugal pump,
  gate/control/check valve, instrument bubble field and shared, heat
  exchanger coil and shell-and-tube, compressor, filter, motor) with OpenCV
  primitives, each with its terminal points written to a JSON sidecar.

- **Detector** — the same synthetic generator as the electronic detector,
  parametrised to point at the P&ID reference directory and use a wider
  symbol-size range (40–260px; the first version topped out too low and
  missed D1's large separator vessel). 900 train / 100 val composites,
  yolov8n, 45 epochs.

- **Tag prior** — `tags.yaml`, the ISA-5.1 convention. PC, LT, FIC and the
  rest → Instrument; FCV, LCV, HCV → Valve_Control; equipment prefixes
  (P → pump, E/H → exchanger, V/D/T → vessel).

- **Specialists** — stubbed, empty, with the pairs that need doing named in
  the docstring (bubble versus exchanger, both circular; vessel aspect
  ratio).

- **`pack.yaml`** ties it together and sets the fusion weights — here OCR
  gets the edge, because an ISA tag that OCR reads cleanly is very strong
  evidence.

### What it does on the real drawings

D1, D2 and D3 route to the P&ID pack and run end to end, producing the JSON
and graph. I hand-built a golden set for D3 the same way I did D4/D5 — ran
the pipeline, rendered an id-only overlay, cross-referenced every box against
the drawing, wrote down 21 real components with their true types.

`edp eval` on D3, with the P&ID pack, no machinery changes:

| | D3 |
|---|---|
| Detection precision | 0.61 |
| **Detection recall** | **0.95** (20 of 21 real components found) |
| Detection F1 | 0.74 |
| Classification (on matched pairs) | 0.55 |

The recall is the number I'd lead with. A detector trained on 900
procedurally-drawn ISA symbols found almost every real component in a P&ID it
never saw. That's the "does the generic approach transfer" question answered
directly.

The two gaps are specific:

- **Precision** — 13 false positives, almost all boxes on descriptive text
  ("Centrifugal Pump", "Air Cooled Exchanger"), on flow arrows, or
  duplicates. The electronic path suppresses text regions before
  localization; wiring a generic OCR-token suppression into the YOLO path and
  tightening the merge removes most of these. It's a machinery change so I'd
  measure and gate it like any other.
- **Classification** — the confusions are almost entirely circular equipment.
  Instrument bubbles (FCV, LCV, HCV) misread as heat exchangers is 3 of the 9
  errors; the accumulator vessel read as an instrument; motors read as
  compressors twice. A bubble-vs-exchanger specialist (internal zigzag versus
  a plain divider or a bare tag) and a motor check ("M" glyph present) are
  exactly the documented-convention case the electronic specialists already
  establish.

The claim here isn't "it's accurate on P&IDs". It's that a genuinely
different drawing type was added as a folder of data plus one 250-line symbol
script, with the eight-stage pipeline, the fusion classifier, the
connectivity graph and the output contract all untouched and still passing
their tests. That's the property the brief asks for, shown rather than
asserted.

---

## 9. Evaluation methodology and the numbers

**Why a golden set at all.** With unlabelled drawings, "validation" started as
self-consistency checks and visual review. Honest, but not falsifiable —
every accuracy claim would be an eyeball estimate. A golden set makes every
number reproducible and disprovable.

**How it's built.** For each drawing: run the pipeline, render an id-only
overlay (boxes and symbol ids, no predicted type, so the true type has to be
read from the actual schematic and not unconsciously copied from the
prediction), cross-reference every detected id against the real drawing,
zoom on anything ambiguous. False positives and duplicates are documented and
excluded so the harness scores them as false positives rather than dropping
them. Missed components are added with estimated coordinates so recall is
scored honestly.

**The harness** (`edp eval`) matches predicted symbols to golden symbols by
IoU (≥0.5, greedy best-first — checked by hand against every matching it
produced), and reports detection precision/recall/F1 plus — only on matched
pairs — classification accuracy and every individual confusion. Matched-only
classification is deliberate: conflating a detection miss with a
classification error makes a bad localizer look like a bad classifier, and
they need different fixes.

**Connectivity** is scored too, when a golden opts in with
`"_connectivity_verified": true` (D4's connections were hand-traced net by
net for this). Two symbols count as connected if they share a net; it's a
symbol-pair precision/recall, computed only over pairs whose both endpoints
were detected, with every missed and spurious pair listed.

**The numbers**, `edp eval` under `domain: auto` so each drawing is routed:

| drawing | domain | detection F1 | classification | connectivity F1 |
|---|---|---|---|---|
| D4 | electronic | 0.914 | 0.812 | 0.641 |
| D5 | electronic | 0.839 | 0.462 | — |
| D3 | P&ID | 0.741 | 0.550 | — |

**The honest story on the electronic classification gap.** Every improvement
was initially measured against D4 alone — the only ground truth that existed
while thresholds were being tuned. D4 climbed to 81%. Building D5's golden
set the same way was the actual test of generalization, and it wasn't fully
there: 46% on the identical architecture. Root-caused, not just reported:
D5's real component labels are legible to a human eye but a specific
font/rendering defeats Tesseract, so the OCR evidence source silently
contributes nothing on D5 even on cases it should trivially resolve. The
combined electronic number (about 65%) is the one I'd defend, because it
isn't quietly overfit to a single drawing.

**Connectivity, measured and improved.** The net-level metric was the last
honest gap. Building it (a hand-traced D4 golden) confirmed the two causes
`docs/08` had diagnosed — bounding-box erasure of nearby wires, and
terminals scaled blindly from a KiCad template landing inside the body. The
fixes (image-derived terminals, ink-aware subtraction) took D4 connectivity
from **F1 0.40 to 0.64, recall 0.28 to 0.63**, with no change to detection
or classification. `docs/13` has the full progression, including the
sub-fix that measurement showed didn't help on its own. Still open:
per-domain ground/rail merging, net-sanity checks, and a P&ID connectivity
golden so the metric covers the second domain.

---

## 9a. The demo

`edp serve` runs a four-panel web view: the input drawing, the
detected-component overlay plus an id/type/connections table, the trimmed
JSON, and a connectivity graph the **browser** builds from that JSON's
`connections` field (a dependency-free force-directed layout, pan/zoom,
components laid out separately with unconnected symbols in a strip below).
The graph is visibly the same data as the netlist — it's derived from the
delivered JSON, not a separate server-side computation. The header shows
the auto-routed domain and per-stage timing.

---

## 10. Things I built, measured, and threw away

Every one of these was fully built and measured. Keeping the discipline to
revert rather than rationalize is, I think, the strongest evidence of
judgment in the project.

| what | what looked good | what I actually measured | verdict |
|---|---|---|---|
| 21-class YOLO retrain (added Optocoupler, Potentiometer, Relay, Load + style variants) | synthetic mAP50 > 0.98 on every new class | real D4 detection F1 **0.914 → 0.686** | reverted same day |
| Mixing a found Roboflow dataset into YOLO training (vetted, 4 of 8 classes matched) | synthetic mAP50 0.912, in line with baseline | real D4+D5 detection F1 **0.879 → 0.174** | reverted within the hour |
| OCR designator-match confidence 0.55 → 0.75 (a plausible recalibration) | — | zero measured change to `edp eval` | reverted — no evidence to keep it |
| Off-the-shelf real-data detector (Roboflow hosted, class-agnostic recall booster) | 28 extra boxes on D4 | detection F1 **0.879 → 0.73–0.84** across confidence thresholds, recall never moved | default off |
| Self-trained class-agnostic detector (yolov8s, ~4300 real circuit images, remapped to one class) | its own held-out mAP50 0.992 | detection F1 **0.879 → 0.66–0.74**, recall never moved | default off |

The last two are worth spelling out because "use an off-the-shelf pretrained
detector" was a real option I explored properly. The idea: the synthetic
detector is the weakest link on unfamiliar styles, so fuse a
detector-trained-on-real-drawings' boxes in as a class-agnostic recall
booster. I built the infrastructure — a Roboflow hosted-inference client with
disk caching and graceful offline degradation, and a local-weights path — and
measured both a hosted model and one I trained myself on a matching-taxonomy
dataset.

Both hurt, for the same reason: recall never moved off 0.906. The synthetic
detector already finds every findable symbol; the three it misses are
genuinely hard (tiny, or overlapping), and a real-data detector can't recover
them — it only adds false positives elsewhere, because it's trained on
hand-drawn and photographed circuits with very different visual statistics
from this project's clean uniform-stroke line art. That's the useful finding:
the domain gap is a property of the target, not of any one model or dataset,
so more effort on real-data detectors isn't the lever. The infrastructure
stays, tested, behind a config flag that's off. It's the fourth and fifth
"measured and shelved" entry, and consistent with the first three.

---

## 11. Limitations, honestly, in priority order

1. **OCR-quality variance across drawing fonts** is the largest currently
   measured source of the electronic D4/D5 gap. Not fixable by more synthetic
   data — it's a rendering-specific Tesseract limitation. Next step is
   text-region-tuned preprocessing or benchmarking an alternative engine,
   gated by the same measure-before-trusting discipline.

2. **P&ID precision (0.61)** — text and arrow false positives. A generic
   OCR-token suppression in the candidate path is the fix, and it's a
   machinery change so it needs measuring across both domains.

3. **P&ID classification (0.55)** — circular-equipment confusion. The
   bubble-vs-exchanger and motor specialists are named and stubbed.

4. **Connectivity recall (D4 0.63)** — measured now, and much better than it
   was, but 15 of 40 pairs still miss. Mostly the 4-pin optocoupler
   crossing the isolation boundary and the battery on the far-right image
   margin. Only D4 has a hand-traced golden; D3/D5 verify type only.

5. **Detector generalization** is the one place trained-model capacity is a
   real, currently-unresolved constraint — demonstrated by the two reverted
   retrains and the two shelved real-data detectors, not just asserted.

6. **The fusion weights** are a reasoned default, not a swept optimum. A
   config-level ablation harness would let me tune them properly; the hooks
   are there (`evidence_weights` is exposed per pack) but the harness isn't.

7. **P&ID connectivity conventions** (dashed = signal, bubble association)
   are scoped, not built.

8. **Preprocessing assumes digitally-generated line art** — no heavy skew,
   perspective, paper texture or scan artifacts, because that was never the
   input.

---

## 12. What I'd do next

In order of value against the rubric:

- **A P&ID connectivity golden** so the net-level metric covers the second
  domain, not just D4 — plus per-domain ground/rail merging (all `Ground`
  symbols → one node) and net-sanity checks.
- **The two P&ID specialists** — bubble-vs-exchanger and motor. Cheap, lifts
  the visible weak number, and demonstrates extending a pack one more time.
- **Generic text-region suppression** in the YOLO candidate path, measured
  across both domains — fixes P&ID precision and can't hurt electronic.
- **The P&ID connectivity conventions** — dashed-line-as-signal and
  instrument-bubble association, both of which slot into stages that already
  exist.
- **A third domain pack** — even a rough one for single-line power diagrams —
  would be the strongest possible demonstration that "add a drawing type =
  add a folder" is real and not a claim that happens to hold for exactly two
  cases.
