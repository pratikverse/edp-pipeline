"""Stage orchestration. This is the only file that knows the pipeline
order (docs/07_project_layout.md) — every stage module takes typed
objects in and returns typed objects out, with no knowledge of neighbours.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from edp.classify.embedder import Embedder
from edp.classify.library import ReferenceLibrary
from edp.classify.match import classify_candidates
from edp.config import Config
from edp.domains.base import DomainPack
from edp.localize import detect_candidates, find_candidates

from edp.preprocess import binarize, to_grayscale, deskew, blue_layer_mask

from edp.ocr import run_ocr, associate_tokens, nearby_token_text

from edp.types import DrawingResult
from edp.validate import validate
from edp.wires import detect_junction_dots, build_nets, skeletonize_wires, subtract_symbols

def run(image_path: str | Path, cfg: Config | None = None) -> tuple[DrawingResult, dict]:
    """Runs the full pipeline on one drawing. Returns (result, timing)."""
    cfg = cfg or Config.load()
    pack = DomainPack.load(cfg.domain)
    image_path = Path(image_path)
    timing: dict[str, float] = {}

    def _stage(name):
        return _Timer(name, timing)

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with _stage("preprocess"):
        gray = to_grayscale(img_bgr)
        binary = binarize(gray, cfg.preprocess)
        binary, _skew_angle = deskew(binary, cfg.preprocess)
        _blue_mask = blue_layer_mask(img_bgr) if cfg.preprocess.color_layer_split else None

    with _stage("text_detect"):
        # OCR runs before localization, not just before association: text
        # glyphs have as many skeleton corners/endpoints as a real symbol,
        # so localization needs the token boxes to exclude them (see
        # localize/proposals.py). Association (id/value assignment) still
        # happens after classification, once symbol bboxes exist.
        tokens = run_ocr(gray, cfg.text)
        # Junction dots also run early for the same reason: a dot is just
        # as "busy" locally as a real symbol, and without excluding it a
        # nearby symbol's candidate box absorbs the dot plus the wire stub
        # leading to it — verified by inspecting D5's R1 crop, which was
        # mostly two junction dots and wire, with only a sliver of the
        # actual resistor. Detected once here, reused unchanged in the
        # wires stage below.
        dots = detect_junction_dots(binary, cfg.wires)

    with _stage("localize"):
        # Experimental branch: a YOLO detector (trained on synthetic
        # composites of our KiCad reference symbols — see
        # scripts/generate_synthetic_dataset.py) replaces the skeleton-
        # density proposer. It only localizes; final type identity still
        # comes from the DINOv2+FAISS match against the reference library
        # in the classify stage below, unchanged (see detect/yolo_detect.py
        # for why). Falls back to the density-based proposer if no trained
        # weights exist yet, or if use_yolo is off.
        if cfg.localize.use_yolo and pack.detector_weights.exists():
            candidates = detect_candidates(img_rgb, cfg.localize, pack.detector_weights)
        else:
            candidates = find_candidates(binary, cfg.localize, text_tokens=tokens, dots=dots)

    with _stage("classify"):
        library = ReferenceLibrary.build(pack.reference_dir, cfg.classify)
        embedder = Embedder(cfg.classify.model)
        # Classification runs before the final text-association stage (see
        # above), but the OCR text-prior evidence source (classify/text_prior.py)
        # still needs to know what's written near each candidate. Reuses the
        # exact same "nearest tokens within max_association_distance" logic
        # associate_tokens uses for symbol.ocr_text_raw, just applied to
        # candidate boxes instead of classified symbols — see
        # text/associate.py's nearby_token_text docstring.
        ocr_hints = [nearby_token_text(c.bbox, tokens, cfg.text) for c in candidates]
        symbols = classify_candidates(
            img_rgb, candidates, library, cfg.classify, pack, embedder, ocr_hints=ocr_hints
        )

    with _stage("text_associate"):
        symbols = associate_tokens(symbols, tokens, cfg.text)

    with _stage("wires"):
        wire_mask = subtract_symbols(binary, symbols)
        skeleton = skeletonize_wires(wire_mask)
        nets = build_nets(skeleton, dots, symbols, cfg.wires)

    with _stage("validate"):
        validation = validate(symbols, nets, cfg.validation)

    result = DrawingResult(
        drawing_id=image_path.stem,
        source_file=image_path.name,
        symbols=symbols,
        nets=nets,
        validation=validation,
        pipeline_version=cfg.output.pipeline_version,
    )
    return result, timing

class _Timer:
    def __init__(self, name: str, sink: dict):
        self.name = name
        self.sink = sink

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink[self.name] = round(time.perf_counter() - self._t0, 4)
        return False
