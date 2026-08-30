# Connectivity — measured and improved

`docs/09` §11.3 named it as the honest gap: `edp eval` scored symbol type
only, so connectivity accuracy was unquantified and the two diagnosed
causes in `docs/08` (§2.5 template-scaled terminals, §2.6 bbox erasure)
had never been verified as the real bottleneck. This closes that.

## 1. A connectivity metric and a hand-traced golden

`edp/eval.py` now scores connectivity when a golden sets
`"_connectivity_verified": true`. Two symbols count as connected if they
share a net (each appears in the other's `connections`); it's a
symbol-pair precision/recall, computed **only over pairs whose both
endpoints were detected** — a missed symbol is a detection failure, not a
connectivity one. The report lists every missed and every spurious pair.

`data/golden/D4.json` — the `connections` were re-traced net by net from
the schematic (they had been inherited from the pipeline's own output, so
scoring against them measured nothing). The one structural correction
worth calling out: **IC1 is an optocoupler**, so the input side (S1, R1,
IC1 pins 1/2, BATT‑13V) is galvanically isolated from the 224 V output
side — the bottom rail is two separate segments, verified at the visible
gap between IC1's pin 2 and pin 4. Confidence ~85 %; the T1/R4 and
R5/gate routing is dense and a couple of terminal assignments are
judgement calls.

## 2. Subtract symbol ink, not the bbox rectangle (docs/08 §2.6)

`wires.subtract_symbols` used to zero the whole bounding‑box rectangle
before skeletonizing. D4's MOSFET box is 148×180 px — any wire routed
near or behind it was erased, silently severing nets. Now each box is
still cleared, but any long straight horizontal/vertical run that entered
one side of the box and left the opposite side — a wire *crossing*, not
the symbol's own body — is restored (detected with a morphological open
using an 11 px line kernel, gated on ink existing just outside both
sides).

## 3. Image‑derived terminals (docs/08 §2.5)

`wires.refine_terminals` (new). The KiCad template still gives the
expected terminal *count* and pin direction, but its normalised pin
fractions are relative to a render that includes the outward pin leads,
so scaled blindly onto the detected box they land inside the body or past
the wire — which is why `terminal_snap_radius` had been loosened 12 → 25
→ 30 px just to catch anything.

Instead: for each of the four box edges, scan a thin band just outside
it, find the ink runs crossing that band (a wire connecting to the
symbol), match each to the nearest template terminal, and snap that
terminal onto the run's midpoint with the edge's outward normal as its
direction. Template terminals with no run nearby are left as the
fallback; runs with no matching template terminal become extra inferred
terminals (a 4‑pin optocoupler matched to a 2‑pin template really does
have 4 connections). With terminals now on the actual wire,
`terminal_snap_radius` was tightened back to 12 and
`terminal_directional_reach` to 25 — a sweep from 8 to 30 px produces the
identical result, confirming the loose value was a symptom.

## Result (`edp eval`, D4)

| | Precision | Recall | **F1** |
|---|---|---|---|
| baseline (bbox erase, template terminals) | 0.688 | 0.282 | **0.400** |
| + ink‑aware subtraction | 0.625 | 0.256 | 0.364 |
| + image‑derived terminals | 0.658 | 0.513 | 0.563 |
| + golden fix (BATT‑13V+ ↔ S1) | 0.658 | 0.625 | **0.641** |

Ink‑aware subtraction on its own slightly *hurt* (recall −1 pair) — the
docs‑hypothesised fix that measurement didn't support alone. But with
image‑derived terminals in place it recovers 3 more real connections
(A/B: F1 0.611 → 0.641), so it's kept. The dominant fix is the
terminals: recall 0.282 → 0.625.

**Detection and classification are unchanged** (D4 F1 0.914 / cls 0.812)
— the change is scoped to the wires stage.

## What still misses (15 pairs on D4)

Mostly IC1 (the 4‑pin optocoupler crossing the isolation boundary) and
BATT‑224V on the far‑right image margin, where one terminal sits at or
past the image edge. Next levers, not yet done: a per‑domain ground/rail
merge (all `Ground` symbols → one node; power rails as named nets), and
net‑sanity checks (flag a net with an implausible terminal count, or one
bridging two rails with no dot). A D3 (P&ID) connectivity golden would
also let the same metric cover the second domain.
