"""Renders data/kicad_raw/*.kicad_sym into data/reference/<class_name>/*.png
with a matching *.terminals.json sidecar (pin name -> point), per
docs/02_model_selection_rationale.md and docs/06_data_model.md.

D4/D5 are validation inputs only — this library is sourced from KiCad, not
from the drawings under test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edp.classify.kicad_import import render_kicad_file  # noqa: E402

# file stem -> (class_name, symbol_name_filter or None for "take all/first")
SOURCES = {
    "Resistor": "Resistor",
    "Capacitor": "Capacitor",
    "Capacitor_Polarized": "Capacitor_Polarized",
    "Inductor": "Inductor",
    "Crystal": "Crystal",
    "Diode": "Diode",
    "LED": "LED",
    "Zener": "Zener",
    "BJT_NPN": "BJT_NPN",
    "BJT_PNP": "BJT_PNP",
    "MOSFET_N": "MOSFET_N",
    "Ground": "Ground",
    "Battery_Cell": "Battery",
    "Switch_SPST": "Switch",
    "Antenna": "Antenna",
    "Transformer": "Transformer",
    "Fuse": "Fuse",
    "R_Potentiometer": "Potentiometer",
    "Optocoupler_4N25": "Optocoupler",
    "Relay_SPDT": "Relay",
    "Load_Block": "Load",
    # Style variants (docs/08 Phase 1.2 / 1.3): same class, a second drawing
    # convention, so the library isn't blind to whichever one a given real
    # drawing happens to use.
    "Battery_MultiCell": "Battery",  # multi-cell (long/short line pairs) vs Battery_Cell's single cell
    "Transformer_Wavy": "Transformer",  # parallel-wavy-line winding vs Transformer's interleaved-dash core
    "Resistor_Box": "Resistor",  # IEC rectangle-box convention vs Resistor's ANSI zigzag -- D4's own
    # resistors are drawn exactly this way, a diagnosed real style gap (docs/08 section 0.1), not a guess
    "Ground_Earth": "Ground",  # multi-bar chassis/earth symbol vs GND's single-bar convention
    "Switch_Push": "Switch",  # momentary pushbutton vs SW_SPST's toggle-lever convention
}


def main() -> None:
    raw_dir = REPO_ROOT / "data" / "kicad_raw"
    out_dir = REPO_ROOT / "data" / "reference"

    for stem, class_name in SOURCES.items():
        src_path = raw_dir / f"{stem}.kicad_sym"
        if not src_path.exists():
            print(f"[skip] {src_path} not found")
            continue

        rendered = render_kicad_file(src_path)
        if not rendered:
            print(f"[warn] no symbols parsed from {src_path}")
            continue

        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for symbol_name, rs in rendered:
            safe_name = symbol_name.replace("/", "_")
            png_path = class_dir / f"{safe_name}.png"
            cv2.imwrite(str(png_path), rs.image)

            terminals_path = class_dir / f"{safe_name}.terminals.json"
            terminals_path.write_text(
                json.dumps(
                    [
                        {"name": name, "point": list(tip), "body": list(body)}
                        for name, tip, body in rs.terminals
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[ok] {class_name}/{safe_name}.png  ({len(rs.terminals)} terminals)")


if __name__ == "__main__":
    main()
