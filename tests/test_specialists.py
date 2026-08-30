"""Geometric confusion-pair specialists (edp/domains/electronic/specialists.py), tested
against this project's own KiCad reference renders — the same crops used
to hand-validate the rules during development (see docs/08_improvement_plan.md
and the conversation record for the fuller empirical validation against
D4's real crops, which isn't repeated here since it'd require the image
fixture)."""
from pathlib import Path

import cv2
import pytest

from edp.domains.electronic.specialists import (
    battery_capacitor_specialist,
    bjt_potentiometer_specialist,
    select_specialist,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"


def _load(class_name: str, filename: str):
    path = REFERENCE_DIR / class_name / filename
    if not path.exists():
        pytest.skip(f"reference image not present: {path}")
    return cv2.imread(str(path))


def test_capacitor_reference_classifies_as_capacitor():
    crop = _load("Capacitor", "C.png")
    ev = battery_capacitor_specialist(crop)
    assert ev.class_scores.get("Capacitor") == 1.0


def test_multicell_battery_reference_classifies_as_battery():
    crop = _load("Battery", "Battery.png")
    ev = battery_capacitor_specialist(crop)
    assert ev.class_scores.get("Battery") == 1.0


def test_potentiometer_reference_has_no_circle():
    crop = _load("Potentiometer", "R_Potentiometer.png")
    ev = bjt_potentiometer_specialist(crop)
    assert ev.class_scores.get("Potentiometer") == 1.0


def test_bjt_npn_reference_finds_a_circle():
    crop = _load("BJT_NPN", "Q_NPN_BCE.png")
    ev = bjt_potentiometer_specialist(crop)
    assert set(ev.class_scores) == {"BJT_NPN", "BJT_PNP"}


def test_empty_crop_abstains():
    import numpy as np

    empty = np.zeros((0, 0, 3), dtype="uint8")
    assert not battery_capacitor_specialist(empty).has_evidence
    assert not bjt_potentiometer_specialist(empty).has_evidence


def test_select_specialist_matches_battery_capacitor_group():
    fn = select_specialist(frozenset({"Battery", "Capacitor_Polarized"}))
    assert fn is battery_capacitor_specialist


def test_select_specialist_matches_bjt_potentiometer_group():
    fn = select_specialist(frozenset({"BJT_NPN", "Potentiometer"}))
    assert fn is bjt_potentiometer_specialist


def test_select_specialist_none_for_unrelated_classes():
    assert select_specialist(frozenset({"Resistor", "LED"})) is None


def test_select_specialist_npn_pnp_not_routed_in_production():
    # npn_pnp_specialist exists and is unit-testable directly, but
    # deliberately isn't reachable via select_specialist -- see
    # specialists.py's select_specialist docstring for the measured
    # real-world reliability finding that justifies this.
    fn = select_specialist(frozenset({"BJT_NPN", "BJT_PNP"}))
    assert fn is not None  # falls back to the broader BJT/Potentiometer group
    from edp.domains.electronic.specialists import npn_pnp_specialist

    assert fn is not npn_pnp_specialist
