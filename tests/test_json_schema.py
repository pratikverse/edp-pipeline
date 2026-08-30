"""Contract tests: emitted JSON conforms to the problem statement's example
schema and is internally consistent (docs/03, docs/07)."""
from edp.emit import to_json_dict
from edp.types import DrawingResult, Net, Symbol, Terminal, ValidationResult

REQUIRED_SYMBOL_FIELDS = {"id", "type", "coordinates", "connections"}


def _sample_result() -> DrawingResult:
    r1 = Symbol(id="R1", type="Resistor", bbox=(10, 10, 30, 20), confidence=0.9,
                terminals=[Terminal(symbol_id="R1", index=0, point=(10, 15)),
                           Terminal(symbol_id="R1", index=1, point=(30, 15))])
    c1 = Symbol(id="C1", type="Capacitor", bbox=(50, 10, 70, 20), confidence=0.85,
                terminals=[Terminal(symbol_id="C1", index=0, point=(50, 15))])
    net = Net(id="N000", terminals=[("R1", 1), ("C1", 0)], confidence=1.0)
    r1.terminals[1].net_id = "N000"
    c1.terminals[0].net_id = "N000"
    return DrawingResult(
        drawing_id="TEST",
        source_file="test.png",
        symbols=[r1, c1],
        nets=[net],
        validation=ValidationResult(),
    )


def test_required_fields_present():
    out = to_json_dict(_sample_result())
    for symbol in out["symbols"]:
        assert REQUIRED_SYMBOL_FIELDS.issubset(symbol.keys())


def test_connections_reference_existing_ids():
    out = to_json_dict(_sample_result())
    ids = {s["id"] for s in out["symbols"]}
    for symbol in out["symbols"]:
        for target in symbol["connections"]:
            assert target in ids


def test_connections_are_symmetric():
    out = to_json_dict(_sample_result())
    by_id = {s["id"]: set(s["connections"]) for s in out["symbols"]}
    for a, targets in by_id.items():
        for b in targets:
            assert a in by_id[b], f"{a} -> {b} but not {b} -> {a}"


def test_coordinates_shape():
    out = to_json_dict(_sample_result())
    for symbol in out["symbols"]:
        assert len(symbol["coordinates"]) == 4


def test_connectivity_eval_scores_pairs():
    """edp/eval.py net-level connectivity scoring."""
    from edp.eval import evaluate

    golden = {
        "_connectivity_verified": True,
        "symbols": [
            {"id": "A", "type": "Resistor", "coordinates": [0, 0, 10, 10], "connections": ["B", "C"]},
            {"id": "B", "type": "Resistor", "coordinates": [20, 0, 30, 10], "connections": ["A"]},
            {"id": "C", "type": "Resistor", "coordinates": [40, 0, 50, 10], "connections": ["A"]},
        ],
    }
    pred = {
        "symbols": [
            {"id": "P1", "type": "Resistor", "coordinates": [0, 0, 10, 10], "connections": ["P2"]},
            {"id": "P2", "type": "Resistor", "coordinates": [20, 0, 30, 10], "connections": ["P1", "P3"]},
            {"id": "P3", "type": "Resistor", "coordinates": [40, 0, 50, 10], "connections": ["P2"]},
        ],
    }
    r = evaluate(golden, pred, "t")
    assert r.conn_scored
    assert r.conn_tp == 1        # A--B found
    assert r.conn_fn == 1        # A--C missed
    assert r.conn_fp == 1        # B--C spurious


def test_connectivity_eval_opt_in():
    from edp.eval import evaluate
    golden = {"symbols": [{"id": "A", "type": "R", "coordinates": [0, 0, 9, 9], "connections": ["B"]}]}
    pred = {"symbols": [{"id": "A", "type": "R", "coordinates": [0, 0, 9, 9], "connections": []}]}
    assert evaluate(golden, pred, "t").conn_scored is False
