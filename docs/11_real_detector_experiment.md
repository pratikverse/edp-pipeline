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
