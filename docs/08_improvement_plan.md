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

### 1.2 Add the missing classes — *~4h*
Add to `data/reference/` and the synthetic generators:
`IC_DIP`, `Optocoupler`, `Potentiometer` (KiCad `R_Potentiometer`), `Relay`
(KiCad 404'd — source from Wikimedia SVG or draw), `Block`/`Load`, plus alternate
style variants for `Transformer` (parallel-wavy-line convention) and `Battery`.

Regenerate synthetic data, retrain. This is the "add a class = data operation, not a
code change" claim from `docs/02` being exercised for real.

**Success criterion:** zero symbols in the golden set have no correct available class.

### 1.3 Domain-randomised rendering — *~1 day*
In `build_reference_from_kicad.py` and `generate_synthetic_dataset.py`:
- stroke width sampled 1–3px (currently fixed)
- rotation jitter ±3–5°
- 1px morphological dilate/erode (simulates rendering-weight variation)
- Gaussian blur σ ∈ [0, 0.8], contrast/gamma jitter

Attacks the KiCad→real gap (**2.4**) for both classifier paths simultaneously.
Requires a retrain (~15 min on GPU now).

**Success criterion:** gap between synthetic-val mAP50 and golden-set detection F1
narrows by ≥ 30%.

### 1.4 Geometric disambiguation for confusion pairs — *~1 day*
For pairs still confused after 1.1–1.3, add deterministic checks that run *only*
when the classifier's top-2 are a known confusion pair with a thin margin:

- **Capacitor vs Battery** — count horizontal line segments and compare lengths.
  Two equal → capacitor. Alternating long/short (≥3) → battery.
- **NPN vs PNP** — locate the arrowhead on the emitter lead; pointing *toward* the
  base = PNP, *away* = NPN.
- **Diode vs Zener** — cathode bar is a plain line (diode) vs. bent ends (zener):
  check for perpendicular stubs at the bar's ends.

Hand-coding is appropriate here specifically because the distinguishing feature is a
**documented drawing convention**, not a learned statistical pattern — the same
reasoning that justified the junction-dot rule in `docs/06`.

**Success criterion:** confusion-pair error rate < 10%.

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
