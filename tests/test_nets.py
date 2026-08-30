"""Synthetic fixtures for the net-building crossing rule — the only place
it can be verified precisely, since we control the input (docs/04, docs/06).
"""
import numpy as np

from edp.config import WiresConfig
from edp.wires import build_nets


def _cross_skeleton(size=21, dot=False):
    """A + shaped skeleton: horizontal line and vertical line crossing at
    the centre. If dot=True, a small filled disk is also placed there."""
    img = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    img[c, :] = 255  # horizontal
    img[:, c] = 255  # vertical
    return img, (c, c)


def test_undotted_crossing_splits_into_two_nets():
    skeleton, _center = _cross_skeleton()
    cfg = WiresConfig()
    nets = build_nets(skeleton, dots=[], symbols=[], cfg=cfg)
    # No dot -> horizontal and vertical strokes must NOT share a net.
    assert len(nets) == 2, f"expected 2 independent nets, got {len(nets)}"


def test_dotted_crossing_merges_into_one_net():
    skeleton, center = _cross_skeleton()
    cfg = WiresConfig()
    # dot coordinate is (x, y) i.e. (col, row)
    dots = [(center[1], center[0])]
    nets = build_nets(skeleton, dots=dots, symbols=[], cfg=cfg)
    assert len(nets) == 1, f"expected 1 merged net at a dotted junction, got {len(nets)}"


def test_t_junction_without_dot_defaults_connected():
    """T-junctions without a dot are not a valid convention in these
    drawings (see docs/05 limitations) — defensively kept connected
    rather than silently dropped."""
    size = 21
    img = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    img[c, :] = 255
    img[: c + 1, c] = 255  # only the upper half -> 3 arms, not 4
    cfg = WiresConfig()
    nets = build_nets(img, dots=[], symbols=[], cfg=cfg)
    assert len(nets) == 1


def test_refine_terminals_snaps_to_wire_attach_point():
    """refine_terminals moves a template terminal onto where a wire
    actually touches the bounding box."""
    import numpy as np
    from edp.types import Symbol, Terminal
    from edp.wires import refine_terminals

    binary = np.zeros((60, 60), np.uint8)
    # a horizontal wire entering the left edge of a box at y=30
    binary[29:31, 0:22] = 255
    # box is x[20..40] y[20..40]; template terminal wrongly placed at centre
    sym = Symbol(id="S", type="Resistor", bbox=(20, 20, 40, 40),
                 terminals=[Terminal(symbol_id="S", index=0, point=(30, 30), source="library")])
    refine_terminals([sym], binary, band=10)
    tx, ty = sym.terminals[0].point
    assert tx == 20 and abs(ty - 30) <= 2          # snapped to the left edge at the wire
    assert sym.terminals[0].direction_deg == 180.0  # outward = left
    assert sym.terminals[0].source == "inferred"


def test_refine_terminals_noop_without_wires():
    import numpy as np
    from edp.types import Symbol, Terminal
    from edp.wires import refine_terminals

    binary = np.zeros((60, 60), np.uint8)
    sym = Symbol(id="S", type="Resistor", bbox=(20, 20, 40, 40),
                 terminals=[Terminal(symbol_id="S", index=0, point=(30, 30), source="library")])
    refine_terminals([sym], binary)
    assert sym.terminals[0].point == (30, 30)  # untouched
