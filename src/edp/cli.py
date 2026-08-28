"""CLI contract per docs/07_project_layout.md:

    edp run data/raw/D5.png --config config/default.yaml --out outputs/
    edp run data/raw/ --out outputs/
    edp build-library data/reference/ --out data/reference/index.npz
    edp serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edp.config import Config


def _cmd_run(args: argparse.Namespace) -> None:
    from edp.emit.graph_out import export_all
    from edp.emit.json_out import to_json_dict
    from edp.pipeline import run

    cfg = Config.load(args.config)
    target = Path(args.image)
    images = sorted(target.glob("*.png")) if target.is_dir() else [target]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        print(f"[edp] processing {image_path} ...")
        result, timing = run(image_path, cfg)
        json_dict = to_json_dict(result)

        json_path = out_dir / f"{result.drawing_id}.json"
        json_path.write_text(json.dumps(json_dict, indent=2), encoding="utf-8")

        graph_paths = export_all(result, out_dir)

        print(
            f"[edp] {result.drawing_id}: {len(result.symbols)} symbols, "
            f"{len(result.nets)} nets, timing={timing}"
        )
        print(f"[edp]   json  -> {json_path}")
        print(f"[edp]   graph -> {graph_paths['png']}")


def _cmd_build_library(args: argparse.Namespace) -> None:
    import numpy as np

    from edp.classify.library import ReferenceLibrary

    cfg = Config.load(args.config)
    library = ReferenceLibrary.build(args.reference_dir, cfg.classify)
    print(f"[edp] built library with {len(library)} entries")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(library) == 0:
        print("[edp] warning: empty library (no images under reference_dir)")
        return
    np.savez(
        out_path,
        embeddings=library._matrix,
        class_names=[e.class_name for e in library.entries],
        rotations=[e.rotation for e in library.entries],
        mirrored=[e.mirrored for e in library.entries],
        source_paths=[e.source_path for e in library.entries],
    )
    print(f"[edp] saved -> {out_path}")


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("edp.web.server:app", host="0.0.0.0", port=args.port, reload=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the pipeline on an image or directory")
    p_run.add_argument("image")
    p_run.add_argument("--config", default="config/default.yaml")
    p_run.add_argument("--out", default="outputs")
    p_run.set_defaults(func=_cmd_run)

    p_lib = sub.add_parser("build-library", help="build the reference embedding library")
    p_lib.add_argument("reference_dir")
    p_lib.add_argument("--config", default="config/default.yaml")
    p_lib.add_argument("--out", default="data/reference/index.npz")
    p_lib.set_defaults(func=_cmd_build_library)

    p_serve = sub.add_parser("serve", help="run the demo web frontend")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
