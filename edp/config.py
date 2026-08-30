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
    thin_stroke_max_width: int = 4
    density_window: int = 25
    density_threshold: int = 3
    density_merge_kernel: int = 9
    candidate_bbox_pad: int = 4
    candidate_merge_overlap: float = 0.5  # union candidates overlapping more than this (relative to the smaller box)
    use_yolo: bool = True  # YOLO localizer vs. the classical skeleton-density fallback
    # NOTE: the trained weights path is domain-specific and lives in the
    # domain pack (edp/domains/<domain>/pack.yaml), not here.
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
    ambiguity_margin_threshold: float = 0.15
    # NOTE: domain-specific knowledge — the reference-image library, the OCR
    # designator table, the linear-probe artifact, and the evidence-fusion
    # weights — lives in the domain pack (edp/domains/<domain>/pack.yaml),
    # not here. This block is only the model + augmentation params that are
    # the same regardless of drawing type.


class TextConfig(BaseModel):
    upscale_factor: int = 3
    orientations: list[int] = [0, 90, 270]
    max_association_distance: int = 45
    # Multiple Tesseract configs run per orientation and merged (dedup by
    # bbox overlap, highest-confidence token kept) rather than one config
    # picked as "the" answer. Measured empirically (docs/08 section 0.1 /
    # OCR robustness follow-up): no single PSM mode wins on both D4 and D5
    # -- PSM 12 helps D4, PSM 6 with the dictionary disabled helps D5 more,
    # merging strictly beats either alone on both (D4 7->8, D5 2->4 known
    # designators recovered) since a real drawing's OCR difficulty varies
    # by font/rendering in ways no single config generalizes across.
    tesseract_configs: list[str] = [
        "--psm 11",
        "--psm 12",
        "--psm 6 -c load_system_dawg=0 -c load_freq_dawg=0",
    ]


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
    domain: str = "electronic"  # which edp/domains/<name>/ pack to load
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
