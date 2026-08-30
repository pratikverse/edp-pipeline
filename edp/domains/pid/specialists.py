"""Geometry confusion-pair specialists for P&ID symbols.

None routed yet. The obvious candidates once measured against a P&ID
golden set: Valve_Gate vs Valve_Control (actuator present above the
bowtie?), Heat_Exchanger vs Instrument (both circular — internal zigzag
vs. horizontal divider line), Vessel-vertical vs Vessel-horizontal
(aspect ratio). Kept as an explicit empty pack so the DomainPack contract
is satisfied and specialists can be added without touching pipeline code.
"""
from __future__ import annotations

CONFUSION_GROUPS: list[frozenset[str]] = []


def select_specialist(candidate_classes: set[str]):
    return None
