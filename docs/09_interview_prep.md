# Interview Prep — Defend This

Quick-reference for live Q&A. Full reasoning is in the published technical
doc and `docs/01`–`08`; this is the compressed version to have in your head,
not to read from.

---

## The one-sentence pitch

"A pipeline that never trusts one signal — every stage is decoupled, every
model is trained without hand-labeled data, and every classification
decision is a weighted vote across five independent sources that can each
abstain, so a wrong answer is traceable to a specific cause, not a
black-box guess."

## If asked "why not just use a VLM for the whole thing?"

- The brief explicitly rewards a pipeline whose decisions can be justified
  individually — a VLM's "here's what I see" is opaque at exactly that
  granularity.
- Failure modes are far harder to characterize: if the VLM mislabels a
  component, there's no lever to pull except a different prompt or a bigger
  model. A pipeline stage has a specific, fixable cause every time.
- Where an LLM *did* help: authoring the OCR designator table once, at
  build time (not called at runtime). Know this distinction cold — it's
  the difference between "didn't use LLMs" and "used them where they
  add value without becoming a dependency."

## If asked "why these specific models?" (rapid-fire ready)

| Choice | One-line why |
|---|---|
| YOLOv8n for localization | Small, fast, trainable on self-generated synthetic labels — no manual annotation needed |
| DINOv2 over CLIP | Dense visual similarity beats image-text alignment for fine geometric distinctions (resistor zigzag vs. inductor coil) |
| Reference-library matching over a fine-tuned classifier | Adding a class = drop in crops, not retrain — this is what "generic and scalable" means concretely |
| 5-source evidence fusion, not a priority order | A fixed "OCR > vision" rule lets one wrong loud source override two right quiet ones; weighted-by-confidence doesn't |
| Classical CV for wires, not a learned model | Deterministic, needs no training data, and the input (clean orthogonal line art) is the textbook case for skeletonization |
| Nets before pairwise connections | A conductor touches N terminals, not 2 — modeling that directly avoids re-deriving grouping during dedupe |

## If asked "what's your accuracy?"

Don't lead with a number — lead with the methodology, then give it:

"Detection F1 is about 0.88 across both hand-verified drawings.
Classification accuracy is 65.5% combined — but the more important number
is that it went 56.2% → 65.5% through measured, independently-verified
improvements, and I can tell you exactly which of those helped and by how
much, because I built a golden-truth evaluation harness first and
refused to trust any change I couldn't measure against it."

If pushed on why it's "only" 65.5%: **D5's real labels are legible to a
human but a specific font defeats Tesseract** — root-caused, not just
observed. This is a genuinely good answer because it shows you dug past
the symptom.

## If asked "tell me about a mistake you caught"

Two ready-made, both with real numbers:

1. **21-class YOLO retrain**: added missing symbol classes, retrained,
   synthetic mAP50 was excellent (>0.98 on new classes). Real detection F1
   dropped 0.914 → 0.686. Reverted same day, kept the failed weights
   documented rather than deleted, and moved the class-coverage win to the
   classification library instead of the detector.
2. **Roboflow dataset mix**: found a real-drawing dataset, vetted its class
   taxonomy by hand (not just trusting class names — visually confirmed
   which of its 8 classes actually matched ours), mixed a proportionate
   sample into training. Synthetic numbers looked fine. Real detection F1
   collapsed 0.879 → 0.174. Reverted within the hour.

Both demonstrate the same discipline: **synthetic validation numbers are
not evidence of real performance — only measurement against held-out real
drawings is**, and that discipline is worth more than either experiment
succeeding would have been.

## If asked "how would this scale to more symbol types / more drawings?"

- New symbol class: source a KiCad reference (or draw one procedurally if
  no KiCad symbol exists — did this for a few structural variants like
  battery cell count), drop it in `data/reference/`, rebuild the FAISS
  index. No retraining required for classification.
- New drawing style: the reference library already holds multiple real
  style variants per class (IEC box vs. ANSI zigzag resistor) specifically
  because the two evaluation drawings already disagreed on convention —
  this is the generalization the architecture is built around, not an
  afterthought.
- Detector generalization is the harder scaling question — it's the one
  place trained-model capacity is a real constraint, honestly documented
  as a limitation, not hidden.

## If asked "what would you do with more time/resources?"

Be precise about *which* resource, because it's not compute:

"Not more GPU time — training runs took minutes, not hours, throughout.
What's actually scarce is **domain-matched ground truth**: only two real
drawings exist to both tune against and measure against, so any single fix
moves the number by several percentage points and it's easy to overfit to
one drawing's quirks — which is exactly what happened once, and got caught
by measuring the second drawing independently. More real, hand-verified,
target-style drawings is the actual unlock, not more engineering time on
the current two."

## Things to have open/ready if a live demo is requested

- `http://localhost:8123` — full pipeline UI (original / detected symbols
  overlay / JSON / graph, side by side)
- `edp eval` output for D4+D5 — the actual measured numbers, not a claim
- One of the reverted-experiment log excerpts, if you want to show the
  measurement discipline rather than just describe it
