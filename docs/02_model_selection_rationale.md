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
