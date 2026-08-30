"""Unit tests for the evidence-fusion architecture (classify/evidence.py,
classify/text_prior.py) — docs/08_improvement_plan.md. Deliberately don't
touch the GPU-backed embedder/library here; those are exercised end-to-end
via `edp eval` against data/golden/, not unit tests."""
from edp.classify.evidence import ClassificationEvidence, fuse_evidence, no_evidence
from edp.classify.text_prior import EvidenceLevel, evidence_from_text, normalize_ocr_text


def test_no_evidence_is_excluded_from_fusion():
    strong = ClassificationEvidence(source="a", class_scores={"Resistor": 1.0}, confidence=0.9)
    empty = no_evidence("b")
    result = fuse_evidence([strong, empty], weights={"a": 1.0, "b": 1.0})
    assert result.top_class == "Resistor"
    assert "b" in result.trace  # abstention is still recorded for explainability


def test_agreement_beats_a_single_strong_disagreement():
    a = ClassificationEvidence(source="a", class_scores={"Battery": 1.0}, confidence=0.6)
    b = ClassificationEvidence(source="b", class_scores={"Battery": 1.0}, confidence=0.6)
    c = ClassificationEvidence(source="c", class_scores={"Capacitor": 1.0}, confidence=0.9)
    result = fuse_evidence([a, b, c], weights={"a": 1.0, "b": 1.0, "c": 1.0})
    assert result.top_class == "Battery"


def test_fusion_with_no_evidence_returns_unknown():
    result = fuse_evidence([no_evidence("a"), no_evidence("b")], weights={})
    assert result.top_class == "Unknown"
    assert result.candidates == []


def test_margin_is_zero_for_a_tie():
    a = ClassificationEvidence(source="a", class_scores={"Battery": 1.0, "Capacitor": 1.0}, confidence=1.0)
    result = fuse_evidence([a], weights={"a": 1.0})
    assert result.margin == 0.0


def test_normalize_strips_ocr_punctuation_noise():
    assert normalize_ocr_text("R 6") == "R6"
    assert normalize_ocr_text("ZD-1") == "ZD1"
    assert normalize_ocr_text("t1bc548") == "T1BC548"


def test_normalize_applies_safe_dollar_to_s_substitution():
    # observed concretely on D4: Tesseract read the switch designator "S1"
    # as "$1" (see classify/text_prior.py's _SAFE_SUBSTITUTIONS docstring)
    assert normalize_ocr_text("$1") == "S1"


def test_exact_part_number_is_level_a_and_overrides_designator_reading():
    ev = evidence_from_text("T2BC547", cfg_path="edp/domains/electronic/designators.yaml")
    assert ev.class_scores == {"BJT_NPN": 1.0}
    assert ev.metadata["level"] == EvidenceLevel.EXACT_PART_NUMBER
    assert ev.confidence > 0.9


def test_clean_designator_is_level_b_moderate_confidence():
    ev = evidence_from_text("R6", cfg_path="edp/domains/electronic/designators.yaml")
    assert ev.class_scores == {"Resistor": 1.0}
    assert ev.metadata["level"] == EvidenceLevel.DESIGNATOR
    assert 0.4 <= ev.confidence <= 0.7


def test_ambiguous_ocr_abstains_rather_than_guessing():
    ev = evidence_from_text("Ty eS 2202", cfg_path="edp/domains/electronic/designators.yaml")
    assert not ev.has_evidence


def test_unrecognised_designator_prefix_abstains():
    # "TP2" (a test point) has a designator shape but no entry in the table
    # -- must not be forced into some other class.
    ev = evidence_from_text("TP2", cfg_path="edp/domains/electronic/designators.yaml")
    assert not ev.has_evidence


def test_no_text_abstains():
    ev = evidence_from_text(None, cfg_path="edp/domains/electronic/designators.yaml")
    assert not ev.has_evidence
    ev2 = evidence_from_text("   ", cfg_path="edp/domains/electronic/designators.yaml")
    assert not ev2.has_evidence
