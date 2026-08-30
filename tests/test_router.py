"""Domain page-router (edp/domains/base.py::route)."""
from __future__ import annotations

from dataclasses import dataclass

from edp.domains.base import route
from edp.types import TextToken


@dataclass
class _Cfg:
    upscale_factor: int = 1
    orientations: tuple = (0,)
    tesseract_configs: tuple = ("--psm 11",)
    max_association_distance: int = 45


def _toks(words):
    return [TextToken(text=w, bbox=(0, 0, 10, 10), confidence=90.0) for w in words]


def test_route_electronic(monkeypatch):
    monkeypatch.setattr("edp.ocr.run_ocr", lambda *a, **k: _toks(["R6", "10K", "C1", "BC547", "uF"]))
    name, scores, _ = route(None, _Cfg())
    assert name == "electronic"
    assert scores["electronic"] > scores["pid"]


def test_route_pid(monkeypatch):
    monkeypatch.setattr("edp.ocr.run_ocr", lambda *a, **k: _toks(["PC1", "LT2", "Vapor", "outlet", "GPM", "pump"]))
    name, scores, _ = route(None, _Cfg())
    assert name == "pid"


def test_route_falls_back_when_thin(monkeypatch):
    monkeypatch.setattr("edp.ocr.run_ocr", lambda *a, **k: _toks(["???", "..."]))
    name, _, _ = route(None, _Cfg(), default="electronic")
    assert name == "electronic"
