"""CLI contract per docs/07_project_layout.md:

    edp run data/validation/D5.png --config config/default.yaml --out outputs/
    edp run data/validation/ --out outputs/
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
    from edp.classify.library import ReferenceLibrary, _cache_signature, _save_cache

    cfg = Config.load(args.config)
    reference_dir = Path(args.reference_dir)
    out_path = Path(args.out)
    # ReferenceLibrary.build() now auto-caches to <reference_dir>/index.npz
    # on any miss (see classify/library.py) -- this command mainly exists
    # to (a) warm that cache ahead of time (e.g. in CI, before the first
    # `edp run`) and (b) support writing the index to a different explicit
    # path via --out. Passing cache_path=out_path makes both paths share
    # one code path instead of this command hand-rolling its own npz
    # writer with a different (missing-signature) schema, which would
    # otherwise silently invalidate itself on the very next `edp run`.
    library = ReferenceLibrary.build(reference_dir, cfg.classify, cache_path=out_path)
    print(f"[edp] built library with {len(library)} entries")

    if len(library) == 0:
        print("[edp] warning: empty library (no images under reference_dir)")
        return
    if not out_path.exists():
        # build() only writes the cache on a miss; if --out already held a
        # valid cache (identical signature) nothing needed writing, but an
        # explicit `edp build-library` call should still guarantee the
        # file exists at the requested path.
        signature = _cache_signature(reference_dir, cfg.classify)
        meta = [(e.class_name, e.rotation, e.mirrored, e.source_path) for e in library.entries]
        _save_cache(out_path, library._matrix, meta, signature)
    print(f"[edp] saved -> {out_path}")


def _cmd_eval(args: argparse.Namespace) -> None:
    from edp.eval import evaluate_set, print_report

    results = evaluate_set(args.golden_dir, args.predicted_dir)
    print_report(results)


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

    p_eval = sub.add_parser("eval", help="score predicted JSON against hand-verified golden JSON (docs/08 Phase 0)")
    p_eval.add_argument("--golden-dir", default="data/golden")
    p_eval.add_argument("--predicted-dir", default="outputs/golden_prep")
    p_eval.set_defaults(func=_cmd_eval)

    p_serve = sub.add_parser("serve", help="run the demo web frontend")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
