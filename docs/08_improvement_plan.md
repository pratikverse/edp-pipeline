# Improvement Plan — Symbol Classification & Connectivity

Production-grade plan to take the pipeline from "localization works, classification
is noisy" to something defensible. Every intervention below is tied to a *diagnosed*
cause with evidence, not a guess, and each has a stated success criterion.

---

## 1. Where we actually are

Measured on the 52-drawing real validation set (`data/validation/`), mixed-model run:

| | D4 + D5 | 50 fresh circuits |
|---|---|---|
| symbols with ≥1 connection | 74% | 60% |
| type = "Unknown" | 6% | 20% |

Localization is in good shape — boxes are tight, false positives are near zero on
D4, and the topology-realistic validation set scores mAP50 0.913. **The bottleneck
is classification**, and secondarily connectivity completeness.

Hand audit of D4 (18 real components) with the current shipped pipeline:
roughly 7 of 16 classifiable symbols get the correct type.

---

## 2. Root causes (diagnosed, with evidence)

### 2.1 We discard a supervised in-domain classifier in favour of an unsupervised out-of-domain one

This is the single largest issue.

`detect/yolo_detect.py` throws away `box.cls` entirely (every candidate is created
with `kind="symbol"`). Classification is then done exclusively by DINOv2 + FAISS
nearest-neighbour against the KiCad reference library.

But the two models have very different qualifications for this job:

| | YOLO class head | DINOv2 + FAISS |
|---|---|---|
| training | **supervised**, on our exact 17 classes | self-supervised on natural photographs |
| domain | **schematic line art** (our own renders) | natural images; line art is far out of distribution |
| signal | per-class logits learned to discriminate | global CLS-token cosine similarity |
| saw a schematic in training? | **yes** | no |

Measured on D4 (YOLO's highest-confidence box per component):

| Component | YOLO (discarded) | DINOv2 (shipped) | Truth |
|---|---|---|---|
| T1BC548 | **BJT_NPN** ✓ 0.53 | BJT_PNP ✗ | NPN |
| T2BC547 | **BJT_NPN** ✓ 0.27 | MOSFET_N ✗ | NPN |
| T3BC557 | **BJT_PNP** ✓ 0.67 | MOSFET_N ✗ | PNP |
| R3 | **Resistor** ✓ 0.62 | Inductor ✗ | Resistor |
| R6 | **Resistor** ✓ 0.17 | LED ✗ | Resistor |
| C1 | **Capacitor_Polarized** ✓ | Battery ✗ | Polarized cap |
| S1 | MOSFET_N ✗ | **Switch** ✓ | Switch |
| T4 / IR540 | Ground ✗ | **MOSFET_N** ✓ | MOSFET |

YOLO: ~10/16 correct. DINOv2: ~7/16. **They fail on different symbols** — which
means a fusion policy should beat either alone, not just "switch to YOLO".

The capacitor-vs-battery swap the reviewer spotted is diagnostic of exactly this:
those two symbols differ by *one geometric detail* (equal-length parallel plates vs.
alternating long/short cell lines). That is precisely what a global embedding
smooths away and a supervised classifier learns.

**Why this vestige exists:** the original rationale in `docs/02` chose DINOv2
because there was no labelled training data. That constraint dissolved the moment
we built the synthetic generator — we now *have* labelled data (self-generated) and
are already training a supervised model on it. The DINOv2-only path predates that.

### 2.2 The merge step discards class information

`localize/merge.py::merge_overlapping` unions overlapping boxes and keeps only
geometry. But duplicate detections carry *different* class predictions at different
confidences — on D4 the same transistor produced `BJT_PNP 0.27` and `BJT_NPN 0.53`,
and **the higher-confidence one is correct**. We currently discard that tiebreaker.

### 2.3 Missing classes guarantee errors

D4 alone contains `IC1` (optocoupler) and `LOAD` (filled block); D5 has a relay; the
50-sample set contains potentiometers. None exist in the 17-class library — so the
model *must* be wrong on them. Some of the observed "nonsense" labels (IC1 → Zener,
LOAD → Fuse/Inductor) are this, not a model failure.

### 2.4 Domain gap in rendering

Both the reference library and the synthetic training data render KiCad symbols with
a single fixed stroke width (`thickness = max(1, int(scale*0.05))`), clean
anti-aliasing, and exact KiCad proportions. Real drawings vary in stroke weight,
proportions, and sometimes convention entirely (the 50 samples draw transformers as
parallel wavy lines, nothing like KiCad's coil loops).

### 2.5 Terminals are template-scaled, not image-derived

`classify/match.py::_instantiate_terminals` scales KiCad's normalised terminal
fractions onto the detected bbox. This assumes bbox ≡ the render's extent — but
YOLO's box bounds the *symbol*, while KiCad's fractions are relative to a render
that includes outward pin leads. The mismatch puts terminals inside the body or past
the wire.

**Evidence:** we had to loosen `terminal_snap_radius` 12 → 25 → 30px to get any
connections at all. That escalation is a symptom, not a tuning success.

### 2.6 Symbol subtraction destroys wires

`wires/skeleton.py::subtract_symbols` zeroes the entire bbox *rectangle*. D4's
MOSFET box is 148×180px — any wire crossing that region is erased, silently breaking
nets that should connect.

---

## 3. Phase 0 — Measurement (prerequisite for everything else)

**We currently cannot measure classification accuracy automatically.** Every number
in this document above came from either synthetic mAP (which doesn't reflect real
performance) or manual eyeballing. That is not a production footing, and it means we
cannot tell whether any change below actually helps.

### 0.1 Golden evaluation set — *~4h*
Hand-label D4, D5, and 15 of the 50 samples (17 drawings) with ground-truth boxes,
types, and connections. Store as `data/golden/<name>.json`.

Use an existing tool (Label Studio / labelImg → YOLO format → convert) rather than
building a labeller.

> This does **not** violate the "no manually-annotated data" principle in `docs/02`.
> That principle is about *training* data — keeping the model generic and retrainable
> from generated data. Evaluation data must be grounded in human truth or every
> accuracy number we report is unfalsifiable.

### 0.2 Metrics harness — `edp eval` — *~4h*
- **Detection**: precision / recall / F1 at IoU ≥ 0.5 (Hungarian matching pred↔truth)
- **Classification**: accuracy on matched pairs + full 17×17 confusion matrix
- **Connectivity**: net-level precision/recall — for each ground-truth connected
  pair, was it found? Plus false-connection rate.
- Emit a summary table + machine-readable JSON.

### 0.3 Regression gate — *~2h*
Freeze baseline metrics in `tests/regression_baseline.json`; fail CI if any metric
drops more than 2%. Prevents the "fixed one drawing, broke three others" pattern.

**Success criterion for Phase 0:** `edp eval` produces a single reproducible number
for detection, classification, and connectivity that we can defend in review.

---

## 4. Phase 1 — Classification

### 1.1 Fuse YOLO's class head with the DINOv2 match — *~1 day* — **highest value**

Carry `box.cls` / `box.conf` through `Candidate`, and make `classify/match.py` a
fusion policy rather than a DINOv2-only lookup:

- Both agree → that class, high confidence
- Disagree, YOLO confident (≥0.5) → prefer **YOLO** (supervised, in-domain)
- Disagree, YOLO weak → prefer DINOv2, flag reduced confidence
- Both weak → `Unknown`

Evaluate three configurations separately against the golden set — YOLO-only,
DINOv2-only, fused — so the choice is measured rather than asserted.

Also fix **2.2**: `merge_overlapping` should keep the highest-confidence class among
merged boxes instead of dropping class entirely.

**Success criterion:** classification accuracy on matched detections ≥ 75%
(from ~44% today).

### 1.2 Add the missing classes — *~4h* — **done 2026-08-29**
Added to `data/reference/` (via `data/kicad_raw/` + `scripts/build_reference_from_kicad.py`):
`Optocoupler` (KiCad `4N25`, the base symbol `4N35` and family `extends` — using
the base directly), `Potentiometer` (KiCad `R_Potentiometer`), `Relay` (KiCad
*does* have `Relay_SPDT` under `Relay.kicad_symdir` — the plan's "KiCad 404'd"
note turned out to be about the wrong subdirectory, not a real gap), `Load`
(no KiCad equivalent exists for a generic labelled block — hand-authored as a
minimal S-expression file in the same convention as the rest of
`data/kicad_raw/`, not sourced), plus style variants `Battery_MultiCell` (the
long/short multi-cell symbol, vs. the existing single-cell `Battery_Cell`) and
`Transformer_Wavy` (parallel-wavy-line winding convention, hand-authored —
KiCad's library only ships part-specific transformer footprints, no generic
wavy-line symbol). Class count: 17 → 21.

Regenerated the mixed synthetic training set (`data/synth_mixed/`: 500 scatter +
300 dense + 150 ladder-topology = 950 train / 60 val images, all classes
dynamically enumerated from `data/reference/` so no generator code changes were
needed) and retrained (`symbol_detector_mixed_v2`, 40 epochs, GPU). This is the
"add a class = data operation, not a code change" claim from `docs/02` exercised
for real — every generator picked the new classes up automatically.

**Success criterion:** zero symbols in the golden set have no correct available
class. **Met** — `data/golden/D4.json`'s two previously-`Unknown` symbols
(`SYM_004` LOAD block, `SYM_012` IC1 MCT2E optocoupler) were updated to `Load`
and `Optocoupler` once those classes existed.

**But retraining YOLO on the expanded 21-class set is NOT shipped as the
default**, based on measurement, not assumption: `symbol_detector_mixed_v2`
scores excellently on synthetic validation (mAP50 > 0.98 on every new class)
but regresses real-world detection on `edp eval`'s D4 golden set —
F1 0.914 → 0.686. Concretely: boxes that the 17-class model found confidently
(Switch, Capacitor_Polarized, Battery) come back from the 21-class model at
conf 0.02–0.10, below any reasonable threshold, and misclassified when they
do surface. Believed cause: yolov8n's fixed ~3M-param capacity spread across
21 classes instead of 17, not anything wrong with the new classes themselves
— the new classes fit *training* data fine, the regression shows up on the
out-of-distribution real image. `config/default.yaml` keeps `yolo_weights`
pointed at the 17-class run; the 21-class weights are kept
(`outputs/yolo_runs/symbol_detector_mixed_v2/`) for whenever this gets fixed
(more epochs, more per-class synthetic data, or a larger backbone — each is a
testable hypothesis, not yet tried).

**Second-order finding**: even with YOLO reverted, `data/reference/`'s
expansion alone shifted two of DINOv2's nearest-neighbour matches on D4 —
`SYM_011` (T2BC547, BJT_NPN) now matches `Potentiometer` and `SYM_012` (IC1)
now matches `Relay`, both wrong, where they matched different wrong classes
before. Classification accuracy on the golden set is unchanged in aggregate
(9/16 correct either way) but the specific confusions moved, which is exactly
why `edp eval` needs to gate a library change like a code change, not just a
model swap — see `docs/08` section 3.

### 1.3 Domain-randomised rendering — *partially done 2026-08-29*
Implemented as `scripts/domain_randomize.py` (rotation jitter ±4°, stroke-width
dilate/erode, Gaussian blur, sensor noise, contrast/gamma jitter, simulated JPEG
re-encode) — semantics-preserving only, no aspect distortion, no rotation past a
few degrees (the existing 0/90/180/270 handling already covers real orientation
changes).

Used for a **linear probe**, not a YOLO retrain: `scripts/train_linear_probe.py`
generates ~7,300 domain-randomised crops (every reference image × rotation ×
mirror × 40 jittered variants) and fits a logistic-regression classifier on their
frozen DINOv2 embeddings. Wired in as a new, independent `dinov2_probe` evidence
source (`classify/probe.py`) alongside the existing nearest-neighbour `dinov2`
source — not a replacement, so a bad probe reading can't override a good
nearest-neighbour one on its own, only outvote it.

Deliberately did **not** retrain YOLO with domain-randomised composites this
pass, given Phase 1.2's YOLO retrain already measurably regressed real detection
despite excellent synthetic-val numbers (section 1.2 above) — the probe route
tests the same "attack the KiCad→real gap" idea with the detector held fixed, so
a bad outcome is isolated to classification and easy to revert (as the geometry
specialists and OCR prior already are).

**Result (measured, `edp eval` on D4, chained on top of the OCR prior and
geometry specialists below):** classification accuracy 75.0% → 81.2% (12/16 →
13/16), zero detection regression. Two symbols flipped correct (`SYM_006`
Battery, `SYM_013` Capacitor_Polarized); no new errors introduced. Caught one
routing gap live: the probe's own vote can push an *unrelated* class to a
narrow top-1 ahead of the true confusion pair sitting one rank down (observed
concretely on `SYM_013`, where `LED` edged out `Battery`/`Capacitor_Polarized`
by a 0.008 margin) — `match.py`'s specialist router was widened to check "≥2 of
top-3 belong to a group" in addition to "top-1 belongs to a group" specifically
because of this.

**Remaining from the original idea:** applying domain randomisation to YOLO's
own training composites (`generate_synthetic_dataset.py`) is still open, still
carries the same regression risk Phase 1.2 already demonstrated, and would need
the same `edp eval`-gated treatment before shipping.

### 1.4 Geometric disambiguation for confusion pairs — *partially done 2026-08-29*
Implemented in `classify/specialists.py`, routed only when the fused top
candidates fall in a known confusion group (`match.py`'s
`_select_ambiguous_specialist`), each specialist free to abstain rather than
force a call:

- **Battery vs Capacitor vs Capacitor_Polarized** — done, shipped. Plate count
  (≥4 alternating → Battery), plate curvature (bowed → Capacitor_Polarized in
  this project's own drawing style), "+" mark as a weaker fallback vote.
- **BJT (either polarity) vs Potentiometer** — done, shipped. Hough-circle
  presence (BJT family, drawn inside a circle) vs. rectangular body with no
  circle (Potentiometer).
- **NPN vs PNP** (arrowhead direction) — implemented and unit-tested, **not
  routed**: 2/2 correct on this project's own clean reference renders but only
  1/3 correct on D4's real transistor crops, worse than chance there. Left in
  the module, documented, and deliberately excluded from
  `select_specialist`'s routing rather than shipped anyway.
- **Diode vs Zener** (bent-cathode-stub check) — not started.

Hand-coding is appropriate here specifically because the distinguishing feature is a
**documented drawing convention**, not a learned statistical pattern — the same
reasoning that justified the junction-dot rule in `docs/06`.

**Success criterion:** confusion-pair error rate < 10%. **Not yet met overall**
(Battery/Capacitor pair is now fully resolved on D4; BJT/Potentiometer and
Diode/Zener are not) — see the conversation record / `edp eval` output for the
current per-symbol breakdown.

### 1.5 Calibrated confidence & abstention — *~4h*
Replace the single global `unknown_similarity_threshold: 0.62` with per-class
thresholds fitted on the golden set. Emit top-3 candidates with confidences in the
extended JSON. Abstain to `Unknown` on thin top-1/top-2 margins.

**Success criterion:** of symbols reported with confidence ≥ 0.8, ≥ 95% are correct
(i.e. confidence means something).

---

## 5. Phase 2 — Connectivity

### 2.1 Image-derived terminals — *~1 day*
Stop scaling KiCad fractions onto the bbox. Instead:
- Use the template only for *expected terminal count and pin names*
- Find actual terminals where wire-skeleton pixels cross the symbol's ink mask boundary
- Match found points to template terminals by angle/position to assign names
- Fall back to template-scaled only when image-derived finds nothing

Should allow `terminal_snap_radius` back down to ~5–8px — tight snapping means far
fewer wrong attachments in dense regions.

**Success criterion:** connectivity recall ≥ 80% on the golden set with
`terminal_snap_radius ≤ 10`.

### 2.2 Subtract symbol ink, not the bbox rectangle — *~4h*
Fixes **2.6**. Build a mask of the symbol's actual ink within the bbox (excluding
long thin runs that are clearly wire passing through) and subtract only that.

**Success criterion:** no ground-truth connection is lost to bbox erasure.

### 2.3 Net validation & over-merge detection — *~4h*
- Flag nets with implausible terminal counts (>6 on drawings this size)
- Enforce the junction-dot rule as a *check*: a net merging two rails with no dot
  between them is likely a crossing error → split and flag
- Surface these in the `validation` block instead of silently emitting

### 2.4 Ground / rail semantics — *~4h*
Resolve `docs/05` open question #1: merge all `Ground` symbols into one logical node
(configurable). Detect power rails and treat them as named nets.

---

## 6. Phase 3 — Production hardening

- **Model artefacts**: version and pin YOLO weights + reference index; stop
  depending on "whatever is currently in `outputs/`"
- **Determinism**: seed everything; identical input → identical output
- **Structured logging**: per-stage timings and counts as JSON lines
- **Config validation**: reject impossible threshold combinations at load time
- **Batch API**: async job endpoint for bulk sets, not one-at-a-time HTTP
- **Graceful degradation**: every stage already degrades rather than crashing —
  add tests that assert this on malformed input
- **Packaging**: `pip install .` + `edp` entrypoint works from a clean checkout,
  with weights fetched or bundled

---

## 7. Sequencing

**Week 1**
| Day | Work |
|---|---|
| 1 | Phase 0.1 + 0.2 — golden set and metrics harness (everything else needs this) |
| 2 | Phase 1.1 — YOLO/DINOv2 fusion + merge class-retention; measure immediately |
| 3 | Phase 1.2 + 1.3 — missing classes, domain randomisation, retrain |
| 4 | Phase 2.2 + 2.1 — connectivity fixes |
| 5 | Phase 1.4 — geometric disambiguation for whatever remains confused |

**Week 2**
Phase 1.5, 2.3, 2.4, then Phase 3 hardening and documentation.

---

## 8. Honest expectations

These are hypotheses with reasoning, not promises:

- **1.1 (fusion) is the big one.** YOLO already gets 10/16 on D4 where the shipped
  path gets 7/16, and the errors are complementary. I'd expect matched-detection
  accuracy to land somewhere in 70–85%. This is the cheapest, highest-confidence win.
- **1.2 (missing classes)** removes a category of guaranteed error rather than
  improving a model — small percentage, but it eliminates the most obviously wrong
  outputs a reviewer will notice first.
- **1.3 (domain randomisation)** is standard practice for synthetic→real transfer and
  should help, but magnitude is genuinely uncertain.
- **Phase 2 (connectivity) is the harder problem** and I am less confident here.
  60% → 80% is a reasonable target; getting past ~90% likely needs work beyond this
  plan (curved wires, multi-page nets, implicit connections).
- **Diode/Zener and NPN/PNP may not fully resolve** even after 1.4 — at D5's
  resolution the distinguishing detail is a handful of pixels. Abstention (1.5) may
  be the honest answer for a residual fraction.

## 9. What this plan deliberately does not do

- **Train on found/real circuit datasets.** No labelled bounding boxes exist for
  them, hand-labelling breaks the generic-pipeline principle, and non-KiCad symbol
  styles would desynchronise the detector from the reference library the classifier
  matches against. Synthetic generation from our own symbol source stays the right
  call — see `docs/02`.
- **Replace the classical CV connectivity stage with a learned model.** The
  junction-dot rule is a documented drawing convention; encoding it structurally
  (`docs/06`) is more robust and more explainable than learning it, and there is no
  connectivity training data.
