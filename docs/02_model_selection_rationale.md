# Model & Technique Selection Rationale

This doc exists to answer the assessment's explicit requirement: be able to
justify why each model/technique was chosen and what the trade-offs were.

## Symbol localization: classical CV (contours/connected components), not a trained detector
**Chosen because:** electrical symbols are enclosed, high-contrast line shapes
on a clean background — exactly the case connected-component analysis handles
well without any labeled data.
**Alternative considered:** fine-tuned object detector (YOLOv8/Faster R-CNN).
**Rejected for this timeframe because:** requires labeled bounding boxes we
don't have, and training/eval would consume most of the 2–3 days budget.
**Trade-off accepted:** classical CV will mis-group touching/overlapping
symbols more often than a trained detector would. Documented as a known
limitation, not silently ignored.

## Symbol classification: pretrained ViT embeddings (DINOv2) + nearest-neighbor
**Chosen because:** electrical symbols are highly standardized (IEC 60617 /
ANSI), so a small reference library per class is sufficient — no fine-tuning
needed. This directly satisfies the "generic, scalable" evaluation criterion:
adding a new symbol type is a data operation (add reference embeddings), not
a retraining operation.
**Alternative considered:** CLIP embeddings.
**Why DINOv2 over CLIP:** DINOv2 is trained for dense visual similarity
(better for near-duplicate/shape matching of technical line-art symbols);
CLIP is trained for image-text alignment and tends to be weaker on fine
geometric distinctions between visually similar symbols (e.g. resistor vs
inductor zigzag/coil variants).
**Trade-off accepted:** embeddings alone cannot disambiguate same-type
instances (R1 vs R2) — resolved separately via OCR (see below), not by the
classifier.

## Instance/value resolution: OCR (Tesseract) + proximity association
**Chosen because:** distinguishing R1 from R2 or reading "10K" vs "180pF" is
a text-recognition problem, not a shape-recognition problem. Keeping this as
a separate pipeline stage (rather than trying to bake it into the vision
model) keeps each component simple and independently testable.
**Trade-off accepted:** OCR misreads on small/rotated labels are a realistic
failure mode; proximity-based association can mis-assign a label if two
symbols are very close together.

## Wire/line detection: skeletonization + Hough/skeleton-graph, not learned segmentation
**Chosen because:** wires in these drawings are thin, high-contrast,
orthogonal-or-straight strokes — the textbook case for classical
skeletonization and line extraction. No training data exists for a learned
segmentation model, and one isn't needed for this input distribution.
**Trade-off accepted:** will struggle on curved/freehand wire routing if
that appears in some drawings — flagged as a limitation rather than
over-engineered against upfront.

## Connectivity inference: proximity snapping + junction-dot template matching
**Chosen because:** it directly encodes the actual drawing convention
(filled dot = connected, bare crossing = not connected) rather than
guessing from geometry alone, which is the single biggest source of wrong
answers if skipped.
**Alternative considered:** treat all crossings as connections (simpler, but
wrong on any drawing with crossing-but-unconnected wires — unacceptable
given the evaluation explicitly scores connectivity correctness).

## Where a VLM is used, and where it deliberately isn't
**Used as:** an optional fallback for ambiguous local regions (dense
junction clusters) where classical detection confidence is low — a targeted
crop + question, not a pipeline-wide dependency.
**Deliberately not used as:** the primary detection/connectivity engine.
Reasoning: the evaluation rewards an explainable, engineered pipeline whose
individual decisions can be justified; a VLM-only "describe this schematic"
approach is opaque, harder to justify component-by-component, and its
failure modes are harder to characterize and fix.

## Rotation handling: augmented reference library, not a rotation-invariant model
**Chosen because:** the provided drawings contain the same symbol at multiple
orientations (R1 vertical vs. R4 horizontal; inductors both ways), and DINOv2
embeddings are **not** rotation-invariant — a rotated resistor lands far from
its upright reference in embedding space. Augmenting the library with
0/90/180/270 rotations plus mirrors costs four extra vectors per reference and
solves it entirely.
**Alternative considered:** a rotation-invariant descriptor (Hu moments, RIFT)
or canonicalising each crop by principal axis before embedding.
**Rejected because:** invariant descriptors are much weaker at separating
visually similar line-art symbols, and axis canonicalisation is unstable for
near-square symbols where the principal axis flips on noise.
**Bonus:** the matched rotation is itself useful output — it orients the
terminal template onto the candidate bbox, which is what makes pin-level
connectivity possible.
**Trade-off accepted:** library size grows 4–8×. Irrelevant at this scale
(hundreds of vectors, exhaustive NN search is microseconds).

## Multiple style variants per class
**Chosen because:** D4 and D5 already disagree on convention — D4 draws BJTs
inside circles, D5 draws them bare. Two drawings, two styles. Any real corpus
will be worse.
**Implication:** the library is keyed by class with several exemplars per
class, and matching is nearest-neighbour over all exemplars, not a per-class
centroid. Centroids would average the circled and bare variants into a
prototype resembling neither.
**Why this strengthens the overall argument:** this is exactly the
generalization case a fine-tuned detector handles badly without retraining,
and that the reference-library design absorbs as a data operation.

## OCR upscaling and multi-orientation passes
**Chosen because:** D5 is 680 px wide and its labels are a few pixels tall.
Tesseract has an effective floor around 20 px of x-height; below it, output is
noise regardless of preprocessing. A 3–4× Lanczos upscale before OCR is the
single highest-impact text-stage decision.
**Multi-orientation because:** D4 sets "LOAD" vertically. A single horizontal
pass silently drops rotated labels rather than failing loudly.
**Alternative considered:** a learned scene-text model (PaddleOCR, TrOCR).
**Position taken:** keep Tesseract as the default — it is deterministic,
dependency-light, and adequate on clean synthetic glyphs once upscaled. The
`text/ocr.py` module is isolated behind a narrow interface specifically so
PaddleOCR can be swapped in if measured results justify it.
**Trade-off accepted:** upscaling reconstructs, it does not add information.
Sub-5px glyphs stay unreliable and remain a documented limitation.

## Colour-layer separation (D5 only)
**Chosen because:** D5 renders some conductors in blue and others in black.
Splitting channels is nearly free and gives a strong prior on net membership.
**Used as:** a cross-check that raises or lowers net confidence in stage 6.
**Deliberately not used as:** a hard dependency — D4 is monochrome, so any
logic that requires colour would fail on half the corpus. This is the general
rule applied throughout: opportunistic signals improve confidence, they never
gate correctness.

## Nets as the primary connectivity object
**Chosen because:** a schematic conductor joins N terminals, not two — the D5
top rail reaches C1, R1 and the IC as one net. Emitting pairwise links
directly from traced segments forces the pipeline to re-derive that grouping
during dedupe, and turns every junction merge into a special case.
**Implementation consequence:** the "crossing without a dot is not a
connection" rule is enforced *structurally*, by splitting degree-4 skeleton
nodes into two pass-through paths before taking connected components, rather
than as a correction applied afterwards. Encoding the drawing convention into
the graph topology is more robust than filtering bad edges out later.
**Trade-off accepted:** one more intermediate representation to explain. Paid
back immediately in stage 7, where "a net with fewer than two terminals" is a
meaningful validation check that pairwise edges cannot express.

## [Experimental branch] Symbol localization: YOLOv8 on synthetic data, replacing skeleton density
**Context:** a manual accuracy audit (comparing pipeline output against D4/D5
by eye, since no ground truth exists — see docs/05) found the skeleton-
density localizer had a ~60% false-positive rate on D4, almost entirely wire
bends and rail junctions misread as symbols, plus missed IC1 and one MOSFET
entirely.
**Chosen because:** a trained detector directly learns "does this look like a
symbol," rather than approximating it through a hand-tuned corner-density
heuristic. The original objection to a trained detector (docs/02, symbol
localization entry above) was the lack of labeled bounding-box data — that
objection is answered, not overridden, by generating the labels ourselves.
**How the "no labeled training data" principle is preserved:**
`scripts/generate_synthetic_dataset.py` composites the existing KiCad-
rendered reference symbols (rotated, scaled to the real drawings' size
distribution, with synthetic wire/text clutter) onto blank canvases and
auto-generates the YOLO bounding-box labels from the placement it chose. No
image was manually annotated; the "labels" are a byproduct of generation, not
human judgment about what's in a drawing.
**Alternative considered:** searching for an existing public schematic-symbol
detection dataset.
**Rejected because:** most available options are hand-drawn-diagram datasets
(a different visual style from D4/D5's clean digital line art), and none
would include the specific symbol variants this pipeline's own reference
library already defines precisely.
**Trade-off accepted:** YOLO's class head is trained purely on synthetic
composites, so its own classification is not trusted as final — every
YOLO-proposed box still goes through the existing DINOv2+FAISS match against
the same KiCad reference library (classify/match.py, unchanged). YOLO's job
is localization only.
**Result, training complete (40/40 epochs, 3.49h CPU, final synthetic-val
mAP50=0.876/mAP50-95=0.768; evaluated on D4/D5 directly throughout, not
just synthetic validation mAP):**
- **D4:** false-positive rate dropped from ~60% to ~0% across every
  checkpoint tested (11/11 at epoch 20, 19/19 at epoch 40 land on real
  components). IC1, the MOSFET, and ZD1 — all missed entirely by the
  density-based localizer — are now localized, and ZD1's type is now also
  correctly matched (Zener). Only C1 and BATT-13V remain unlocalized.
- **D5:** recall was the weak point early (4/~25 at epoch 20, default
  conf=0.25) — two separable causes, both addressed: (a) the confidence
  threshold was cutting off real detections sitting in the 0.1-0.25 range
  (verified directly by sweeping conf 0.25/0.1/0.05 against the raw
  detector — confirmed as recall bottleneck, not precision protection;
  default lowered to 0.12), and (b) D5 packs symbols more densely than the
  synthetic canvases did. After (a) alone: 15/~25 at epoch 22. After full
  training at the lower threshold: 28 detections, now covering nearly the
  entire drawing (some may be duplicates/overlaps rather than pure false
  positives — not yet disambiguated).
- **Per-class detector quality (final synthetic val mAP50):** most classes
  are strong (>0.93: Antenna, Battery, Capacitor(s), Crystal, Fuse,
  Ground, LED, MOSFET_N, Transformer, Resistor, Switch, Inductor). Weak:
  BJT_NPN (0.51), BJT_PNP (0.60), Diode (0.66), Zener (0.45) — exactly the
  sibling-pairs that differ by one small visual detail (arrow direction
  for NPN/PNP, a small kink for Diode/Zener), matching the confusion
  already observed in manual audits of the classify stage. This reads as
  a genuine fine-grained-discrimination limit at this training scale,
  not a bug.
- **Queued, not yet done:** a denser-packing synthetic variant
  (`generate_synthetic_dataset.py --symbols-min 15 --symbols-max 30`) is
  generated and visually validated but not yet used for a retrain — the
  threshold fix already closed most of the D5 recall gap it was meant to
  address, so its marginal value is unconfirmed.

## [Experimental branch] Classification retrieval: FAISS index, 768-dim DINOv2-base
**Chosen because:** the reference-library design already commits to "add a
class = drop in crops and rebuild the index" (docs/06, docs/07) — at the
current library size (~140 vectors) a brute-force `matrix @ query` scan is
already sub-millisecond, so FAISS buys nothing today, but it buys the same
interface at a size that would matter later (many classes x rotation/mirror
variants x multiple exemplars), for free. Switching `IndexFlatIP` to an
approximate index (IVF/HNSW) if the library ever outgrows exact search is a
one-line change under this design, not a rewrite.
**DINOv2-base (768-dim) over DINOv2-small (384-dim):** a larger embedding
model, on the same reasoning as the original DINOv2-over-CLIP choice —
better separation of visually similar technical line-art symbols, at
roughly double the compute cost, which is affordable at this library size.
**Trade-off accepted:** larger model means slower classification per crop
(measured: DINOv2-base roughly 2-3x DINOv2-small's per-image cost on CPU).
Acceptable given classification isn't the pipeline's throughput bottleneck.
