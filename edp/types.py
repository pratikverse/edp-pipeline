"""Core typed objects passed between pipeline stages.

Every stage in edp.pipeline takes these in and returns these out — no stage
knows what ran before or after it (see docs/07_project_layout.md). This is
the one file every stage module is allowed to import from the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

BBox = tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), pixel space
Point = tuple[int, int]

TerminalSource = Literal["library", "inferred", "ocr"]
LabelSource = Literal["classification", "ocr", "manual"]


@dataclass
class Terminal:
    """A connection point on a symbol.

    Coordinates are in the drawing's pixel space, already transformed from
    the reference library's normalised crop coordinates (or inferred from
    wire-skeleton contact points when no library template exists).
    """

    symbol_id: str
    index: int
    point: Point
    name: Optional[str] = None
    source: TerminalSource = "inferred"
    net_id: Optional[str] = None
    # Outward pin direction in degrees (image space: 0=+x/right, 90=+y/down),
    # None when unknown (e.g. inferred terminals). Used by wires/nets.py for
    # directional snapping instead of a blind circular search — see
    # docs/06_data_model.md.
    direction_deg: Optional[float] = None


@dataclass
class Candidate:
    """A localized region proposed by stage 2, before classification."""

    bbox: BBox
    kind: Literal["symbol", "wire", "ambiguous"] = "ambiguous"
    mask: object = None  # optional binary mask, same shape as bbox crop
    # YOLO's own class prediction, when the localizer is detect/yolo_detect.py
    # (None for the density-based localizer). Carried through so classify/match.py
    # can fuse it with the DINOv2+FAISS match rather than discarding it — see
    # docs/08_improvement_plan.md section 2.1: YOLO is supervised and in-domain
    # (trained on our own schematic symbols), DINOv2 is not, and on D4 they get
    # ~10/16 vs ~7/16 correct with non-overlapping error sets.
    yolo_class: Optional[str] = None
    yolo_confidence: Optional[float] = None
    # Which localizer proposed this box: "synthetic_yolo" (the in-domain
    # detector, carries yolo_class/confidence), "real_detector" (an
    # off-the-shelf detector trained on real circuit photos, used
    # class-agnostically as a recall booster — see edp/realdetect.py), or
    # "density" (the classical fallback proposer).
    source: str = "synthetic_yolo"


@dataclass
class Symbol:
    """A classified circuit component."""

    id: str
    type: str
    bbox: BBox
    confidence: float = 0.0
    value: Optional[str] = None
    rotation: int = 0  # degrees; orientation the reference library matched
    terminals: list[Terminal] = field(default_factory=list)
    label_source: LabelSource = "classification"
    ocr_text_raw: Optional[str] = None
    # Evidence-fusion output (classify/evidence.py), for explainability and
    # ambiguity routing — not part of the trimmed 4-field JSON schema
    # (emit/json_out.py), only the extended debug view. `top_k` is
    # (class_name, fused_score) sorted descending; `margin` is top-1 minus
    # top-2 on a 0-1 scale (1.0 when only one candidate had any evidence).
    # `evidence_trace` is {source_name: {class_scores, confidence, weight,
    # metadata}} — literally what each source said and how much it counted.
    top_k: list[tuple[str, float]] = field(default_factory=list)
    margin: float = 1.0
    evidence_trace: dict = field(default_factory=dict)


@dataclass
class TextToken:
    """A single OCR-recognised text fragment, prior to symbol association."""

    text: str
    bbox: BBox
    confidence: float
    orientation: int = 0  # degrees the page was rotated for this pass


@dataclass
class Net:
    """An equivalence class of terminals joined by one continuous conductor.

    Primary connectivity object (see docs/06_data_model.md). Pairwise
    `connections` in the emitted JSON are a projection of this, not built
    directly from traced wire segments.
    """

    id: str
    terminals: list[tuple[str, int]] = field(default_factory=list)  # (symbol_id, terminal_index)
    polyline: list[list[Point]] = field(default_factory=list)
    junctions: list[Point] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ValidationResult:
    unattached_terminals: list[tuple[str, int]] = field(default_factory=list)
    isolated_symbols: list[str] = field(default_factory=list)
    single_terminal_nets: list[str] = field(default_factory=list)
    low_confidence_symbols: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class DrawingResult:
    """Final assembled output of a full pipeline run on one drawing."""

    drawing_id: str
    source_file: str
    symbols: list[Symbol] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)
    validation: ValidationResult = field(default_factory=ValidationResult)
    pipeline_version: str = "0.1.0"
