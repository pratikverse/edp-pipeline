"""Associate OCR tokens to the nearest symbol bbox; split combined tokens
("R1 10K" -> id + value). See docs/01 stage 4."""
from __future__ import annotations

import re

from edp.config import TextConfig
from edp.types import BBox, Symbol, TextToken

_ID_RE = re.compile(r"^([A-Za-z]{1,4}\d{1,3})$")
_VALUE_RE = re.compile(r"^\d+(\.\d+)?\s*[a-zA-Zµ]*$")


def associate_tokens(symbols: list[Symbol], tokens: list[TextToken], cfg: TextConfig) -> list[Symbol]:
    """Mutates and returns `symbols` with ocr_text_raw/value/id assigned
    from the nearest token within max_association_distance.

    Instance ids ("R1" vs "R2") are assigned by *global* greedy nearest-
    match, not independently per symbol: with density-based localization,
    several candidate bboxes can cluster around one physical label, and an
    independent per-symbol "closest token" lookup lets two different
    symbols both claim the same id ("R4" assigned twice), which corrupts
    every downstream `connections` reference. Greedy assignment consumes
    each id token at most once.
    """
    for symbol in symbols:
        nearby = [t for t in tokens if _distance(symbol.bbox, t.bbox) <= cfg.max_association_distance]
        if nearby:
            nearby.sort(key=lambda t: _distance(symbol.bbox, t.bbox))
            symbol.ocr_text_raw = " ".join(t.text for t in nearby[:3])

    _assign_ids_greedily(symbols, tokens, cfg)
    _assign_values(symbols, tokens, cfg)
    return symbols


def _assign_ids_greedily(symbols: list[Symbol], tokens: list[TextToken], cfg: TextConfig) -> None:
    candidates = []  # (distance, symbol, id_text)
    for symbol in symbols:
        for token in tokens:
            dist = _distance(symbol.bbox, token.bbox)
            if dist > cfg.max_association_distance:
                continue
            for part in token.text.split():
                if _ID_RE.match(part):
                    candidates.append((dist, symbol, part))

    candidates.sort(key=lambda c: c[0])
    claimed_symbols: set[int] = set()
    claimed_ids: set[str] = set()
    for _dist, symbol, id_text in candidates:
        if id(symbol) in claimed_symbols or id_text in claimed_ids:
            continue
        symbol.id = id_text
        symbol.label_source = "ocr"
        for terminal in symbol.terminals:
            terminal.symbol_id = id_text
        claimed_symbols.add(id(symbol))
        claimed_ids.add(id_text)


def _assign_values(symbols: list[Symbol], tokens: list[TextToken], cfg: TextConfig) -> None:
    # Values are not required to be unique (several inductors can share
    # "50uH"), so this stays a simple per-symbol nearest-match.
    for symbol in symbols:
        nearby = [t for t in tokens if _distance(symbol.bbox, t.bbox) <= cfg.max_association_distance]
        nearby.sort(key=lambda t: _distance(symbol.bbox, t.bbox))
        for token in nearby:
            for part in token.text.split():
                if _VALUE_RE.match(part) and not _ID_RE.match(part) and symbol.value is None:
                    symbol.value = part


def _distance(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    # edge-to-edge gap, not centre-to-centre, so a large symbol doesn't
    # unfairly penalise a nearby label
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    if dx == 0 and dy == 0:
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    return (dx ** 2 + dy ** 2) ** 0.5
