"""Typed config load/validate. See config/default.yaml for the values."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    binarize_block_size: int = 31
    binarize_c: int = 7
    denoise: bool = True
    denoise_min_speckle_area: int = 2
    deskew_max_angle_deg: float = 5
    color_layer_split: bool = True


class LocalizeConfig(BaseModel):
    min_component_area: int = 80
    max_component_area: int = 40000
    dash_close_kernel: int = 5
    thin_stroke_max_width: int = 4
    density_window: int = 25
    density_threshold: int = 3
    density_merge_kernel: int = 9
    candidate_bbox_pad: int = 4
    candidate_merge_overlap: float = 0.5  # union candidates overlapping more than this (relative to the smaller box)
    use_yolo: bool = True  # experimental branch: YOLO localizer instead of skeleton-density
    yolo_weights: str = "outputs/yolo_runs/symbol_detector_mixed/weights/best.pt"  # 17-class
    # model, still the shipped default — see config/default.yaml for why the 21-class
    # symbol_detector_mixed_v2 retrain (docs/08 Phase 1.2) isn't swapped in yet despite strong
    # synthetic-val numbers: it measurably regresses real-world (D4) detection via `edp eval`.
    yolo_conf_threshold: float = 0.12  # lowered from 0.25: verified on D5 that most of the
    # detections between 0.1-0.25 confidence land on real components, not noise — the
    # higher threshold was a recall bottleneck, not a precision safeguard (see docs/02)
    yolo_iou_threshold: float = 0.45


class ClassifyConfig(BaseModel):
    model: str = "dinov2_vitb14"  # 768-dim (DINOv2-base); see docs/02
    embedding_dim: int = 768
    rotations: list[int] = [0, 90, 180, 270]
    mirror: bool = True
    unknown_similarity_threshold: float = 0.62
    reference_dir: str = "data/reference"
    # Evidence-fusion architecture (docs/08_improvement_plan.md, classify/evidence.py):
    # each source (yolo, dinov2, text_prior) contributes weight * confidence * score
    # per class; these are the per-source base weights. See classify/match.py's
    # DEFAULT_EVIDENCE_WEIGHTS for the rationale behind the yolo/dinov2 split.
    # Empty here means "use the code defaults" — set entries to override individually.
    evidence_weights: dict[str, float] = {}
    # OCR reference-designator / part-number prior table (classify/text_prior.py).
    reference_designators_path: str = "config/reference_designators.yaml"
    # Below this top-1/top-2 fusion margin, a candidate is "ambiguous" — reserved
    # for confusion-pair geometry specialist routing (docs/08), unused until one
    # exists; kept here now so the threshold is configurable from day one rather
    # than added later as a breaking config change.
    ambiguity_margin_threshold: float = 0.15
    # Linear-probe evidence source (docs/08 Phase 7, classify/probe.py). Points
    # at the artifact scripts/train_linear_probe.py writes; if the file doesn't
    # exist yet, the source just abstains (no_evidence) rather than the
    # pipeline failing -- same graceful-degradation pattern as an empty
    # reference library.
    probe_model_path: str = "outputs/probe/linear_probe.joblib"


class TextConfig(BaseModel):
    upscale_factor: int = 3
    orientations: list[int] = [0, 90, 270]
    max_association_distance: int = 45
    tesseract_config: str = "--psm 11"


class WiresConfig(BaseModel):
    junction_dot_min_radius: int = 2
    junction_dot_max_radius: int = 5
    junction_dot_fill_ratio: float = 0.6  # min fraction of the local disk that must be ink
    terminal_snap_radius: int = 30  # isotropic fallback for terminals with no known direction
    terminal_directional_reach: int = 45  # px; search distance along a known pin direction
    terminal_snap_cone_deg: float = 70  # full cone width around the pin's direction
    skeleton_prune_min_length: int = 3


class ValidateConfig(BaseModel):
    min_confidence: float = 0.5


class OutputConfig(BaseModel):
    outputs_dir: str = "outputs"
    pipeline_version: str = "0.1.0"


class Config(BaseModel):
    preprocess: PreprocessConfig = PreprocessConfig()
    localize: LocalizeConfig = LocalizeConfig()
    classify: ClassifyConfig = ClassifyConfig()
    text: TextConfig = TextConfig()
    wires: WiresConfig = WiresConfig()
    validation: ValidateConfig = ValidateConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def load(cls, path: str | Path = "config/default.yaml") -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
