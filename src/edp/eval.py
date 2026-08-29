"""Phase 0 measurement harness (docs/08_improvement_plan.md section 3).

Compares a pipeline JSON prediction against a hand-verified golden JSON
(data/golden/*.json) and reports real, reproducible numbers instead of
ad-hoc spot-checking:

  - Detection: precision/recall/F1 via greedy IoU matching (Hungarian-free,
    since golden sets here are small enough that greedy-by-best-IoU never
    diverges from optimal in practice).
  - Classification: accuracy on matched pairs only (a detection miss is a
    detection problem, not a classification one -- conflating them would
    make a bad localizer look like a bad classifier), plus a confusion
    list of the actual (predicted, true) mismatches.

Deliberately NOT included yet: connectivity precision/recall. The current
golden files inherit their `connections` from the pipeline's own output
rather than independently hand-traced ground truth (see the `_notes`
field in data/golden/D4.json) -- scoring against that would just be
comparing the pipeline to itself. Wiring this up is real work for when
connections are properly hand-verified (docs/08 Phase 2 territory).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

IOU_MATCH_THRESHOLD = 0.5


@dataclass
class EvalResult:
    drawing_id: str
    num_golden: int
    num_predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    classification_correct: int
    classification_total: int
    confusions: list[tuple[str, str, str]] = field(default_factory=list)
    # (golden_id, predicted_type, true_type) for every matched pair that disagrees
    unmatched_predicted_ids: list[str] = field(default_factory=list)
    unmatched_golden_ids: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def classification_accuracy(self) -> float:
        return self.classification_correct / self.classification_total if self.classification_total else 0.0


def _iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(golden: dict, predicted: dict, drawing_id: str = "") -> EvalResult:
    """Greedy best-IoU matching, highest IoU pairs claimed first, each
    golden/predicted symbol used at most once."""
    golden_symbols = golden["symbols"]
    pred_symbols = predicted["symbols"]

    pairs = []
    for gi, g in enumerate(golden_symbols):
        for pi, p in enumerate(pred_symbols):
            iou = _iou(g["coordinates"], p["coordinates"])
            if iou >= IOU_MATCH_THRESHOLD:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda t: t[0], reverse=True)

    matched_g: set[int] = set()
    matched_p: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, pi in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        matches.append((gi, pi))

    tp = len(matches)
    fp = len(pred_symbols) - len(matched_p)
    fn = len(golden_symbols) - len(matched_g)

    correct = 0
    confusions = []
    for gi, pi in matches:
        g_type = golden_symbols[gi]["type"]
        p_type = pred_symbols[pi]["type"]
        if g_type == p_type:
            correct += 1
        else:
            confusions.append((golden_symbols[gi]["id"], p_type, g_type))

    unmatched_pred_ids = [pred_symbols[pi]["id"] for pi in range(len(pred_symbols)) if pi not in matched_p]
    unmatched_gold_ids = [golden_symbols[gi]["id"] for gi in range(len(golden_symbols)) if gi not in matched_g]

    return EvalResult(
        drawing_id=drawing_id,
        num_golden=len(golden_symbols),
        num_predicted=len(pred_symbols),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        classification_correct=correct,
        classification_total=tp,
        confusions=confusions,
        unmatched_predicted_ids=unmatched_pred_ids,
        unmatched_golden_ids=unmatched_gold_ids,
    )


def evaluate_set(golden_dir: str | Path, predicted_dir: str | Path) -> list[EvalResult]:
    """Evaluates every `<id>.json` in golden_dir against the same-named
    file in predicted_dir. Skips (with a printed warning) any golden file
    with no matching prediction, rather than failing the whole run."""
    golden_dir, predicted_dir = Path(golden_dir), Path(predicted_dir)
    results = []
    for golden_path in sorted(golden_dir.glob("*.json")):
        pred_path = predicted_dir / golden_path.name
        if not pred_path.exists():
            print(f"[edp eval] warning: no prediction for {golden_path.name} at {pred_path}, skipping")
            continue
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        predicted = json.loads(pred_path.read_text(encoding="utf-8"))
        results.append(evaluate(golden, predicted, drawing_id=golden_path.stem))
    return results


def print_report(results: list[EvalResult]) -> None:
    if not results:
        print("[edp eval] no results to report")
        return

    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)
    total_correct = sum(r.classification_correct for r in results)
    total_matched = sum(r.classification_total for r in results)

    print("=" * 72)
    print(f"{'drawing':<12}{'golden':>8}{'pred':>8}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}{'cls_acc':>9}")
    for r in results:
        print(
            f"{r.drawing_id:<12}{r.num_golden:>8}{r.num_predicted:>8}{r.true_positives:>6}"
            f"{r.false_positives:>6}{r.false_negatives:>6}{r.precision:>8.3f}{r.recall:>8.3f}"
            f"{r.f1:>8.3f}{r.classification_accuracy:>9.3f}"
        )
    print("-" * 72)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0
    overall_acc = total_correct / total_matched if total_matched else 0.0
    print(
        f"{'TOTAL':<12}{sum(r.num_golden for r in results):>8}{sum(r.num_predicted for r in results):>8}"
        f"{total_tp:>6}{total_fp:>6}{total_fn:>6}{overall_p:>8.3f}{overall_r:>8.3f}{overall_f1:>8.3f}{overall_acc:>9.3f}"
    )
    print("=" * 72)

    print("\nClassification confusions on matched (predicted != golden) pairs:")
    any_confusion = False
    for r in results:
        for golden_id, predicted_type, true_type in r.confusions:
            any_confusion = True
            print(f"  [{r.drawing_id}] {golden_id}: predicted={predicted_type!r} true={true_type!r}")
    if not any_confusion:
        print("  (none)")

    print("\nFalse negatives (golden symbols with no matching detection):")
    any_fn = False
    for r in results:
        for gid in r.unmatched_golden_ids:
            any_fn = True
            print(f"  [{r.drawing_id}] {gid}")
    if not any_fn:
        print("  (none)")

    print("\nFalse positives (detections with no matching golden symbol):")
    any_fp = False
    for r in results:
        for pid in r.unmatched_predicted_ids:
            any_fp = True
            print(f"  [{r.drawing_id}] {pid}")
    if not any_fp:
        print("  (none)")
