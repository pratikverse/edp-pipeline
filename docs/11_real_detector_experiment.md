# Real-data second localizer — built, measured, shelved

Fourth entry in the "measured and not shipped" ledger (after the two YOLO
retrains and the OCR-confidence recalibration in `docs/09` §9).

## Hypothesis

The synthetic-trained YOLO detector is "only as good as its synthetic
training data" (`docs/09` §2) and is the weakest link on unfamiliar
real-world symbol styles. An off-the-shelf detector trained on *real*
circuit drawings should see line-weight variation, drawing noise and
conventions the synthetic one never did — so fusing its boxes
class-agnostically (take boxes, ignore its taxonomy) into the candidate
list should raise detection recall with no risk to classification.

## Implementation

`edp/realdetect.py` — Roboflow hosted inference
(`jonathanapps/circuit-component-detection/2`, ~2700 real circuit images).
Boxes only, no class. Merged into the same candidate list via
`merge_overlapping`; a box the real detector alone finds is still
classified normally by DINOv2 + probe + OCR. Responses cached to disk;
any failure (offline / no key) logs once and returns `[]`. Declared per
domain in `edp/domains/electronic/pack.yaml`; toggled by
`localize.use_real_detector`.

## Result (`edp eval`, D4 + D5 combined)

| real_detector | boxes added (D4/D5) | Precision | Recall | **Detection F1** | Classification |
|---|---|---|---|---|---|
| **off (shipped)** | — | 0.853 | 0.906 | **0.879** | 0.655 |
| on, conf 0.30 | 28 / 7 | 0.604 | 0.906 | 0.725 | 0.690 |
| on, conf 0.45 | — | 0.659 | 0.906 | 0.763 | 0.655 |
| on, conf 0.60 | — | 0.744 | 0.906 | 0.817 | 0.655 |
| on, conf 0.75 | — | 0.784 | 0.906 | 0.841 | 0.655 |

**Recall never moved.** At every confidence threshold the real detector
added only false-positive boxes and recovered none of the 3 symbols the
synthetic detector misses. Detection F1 is strictly worse than baseline
for every setting tried.

## Why it failed

The model was trained on real hand-drawn / photographed circuits, which
carry very different visual statistics from this project's clean,
uniform-stroke KiCad-rendered line art — the same domain-gap direction
that sank the Roboflow-dataset training experiment in `docs/09` §9, just
approached from the other side. It fires readily on wire junctions,
label text and border lines that don't look like symbols to a detector
tuned for schematic line art.

## Follow-up: train our own class-agnostic detector (docs/09 §9 "Option C")

Since no circuit detector on Roboflow Universe publishes downloadable
weights, the alternative was to train one. `scripts/build_realdata_detector.py`
takes `nadim-ahmed/circuit-component-detection` v21 (~4300 real circuit
*schematic* images, a 17-class taxonomy that closely matches ours),
remaps every class to a single `symbol` (dropping `Wire_Overlap`), and
trains `yolov8s single_cls` — 30 epochs, GPU. Its own held-out val:
mAP50 0.992. Wired in via `edp/realdetect.py` `provider: local`.

### Result (`edp eval`, D4 + D5 combined)

| conf | boxes added (D4/D5) | Precision | Recall | **Detection F1** | Classification |
|---|---|---|---|---|---|
| **off (shipped)** | — | 0.853 | 0.906 | **0.879** | 0.655 |
| 0.25 | 19 / 31 | 0.518 | 0.906 | 0.659 | 0.690 |
| 0.40 | 19 / 28 | 0.537 | 0.906 | 0.674 | 0.690 |
| 0.55 | 18 / 24 | 0.580 | 0.906 | 0.707 | 0.690 |
| 0.70 | 15 / 19 | 0.630 | 0.906 | 0.744 | 0.690 |

**Same outcome, and the same root cause.** Recall never moved off 0.906 —
the synthetic detector already finds every findable symbol; the 3 it
misses are genuinely hard (tiny / overlapping estimated boxes) and a
real-data detector can't recover them, it only adds false positives
elsewhere. D4 alone at conf 0.70 nearly recovers baseline (F1 0.889) but
D5 collapses (0.619) — the detector fires heavily on D5's specific
rendering. Classification ticks up (+3.5pt) only because a couple of the
spurious boxes happen to land on real symbols and get classified; with
17–27 false positives that is not a win.

**Two independent attempts — off-the-shelf and self-trained, on
different real datasets — both measurably hurt, for the same reason.**
That is the useful finding: the KiCad-line-art vs. real-drawing gap is a
property of the target, not of any one model or dataset, so more effort
on real-data detectors is not the lever. The synthetic detector's
weakness is genuine but the fix is better synthetic coverage (docs/09
§9), not a real-data crutch that trades precision for nothing.

## Decision

Default **off**. Infrastructure kept and tested (`tests/test_realdetect.py`),
pack declaration kept, so `--config` with `use_real_detector: true`
reproduces the numbers above. This is a config flag, not a model swap —
the class-agnostic, additive design means enabling or disabling it can
never touch the shipped path.

## What would actually be worth trying next

A detector trained on `nadim-ahmed/circuit-component-detection-9wiz7`
(~8900 images, a 17-class taxonomy that closely matches ours, includes a
`Wire_Overlap` class) — remapped to a single class and trained locally as
a class-agnostic proposer. No hosted model exists for it, so this is the
"train your own" route, gated by the same `edp eval` check. Not done here
because the brief's Option B was specifically "off-the-shelf pretrained".
