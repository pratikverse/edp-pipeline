"""Thin FastAPI wrapper around edp.pipeline.run(). Holds no pipeline logic
of its own — see docs/07_project_layout.md ("Demo frontend")."""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from edp.config import Config
from edp.emit.graph_out import export_all
from edp.emit.json_out import to_json_dict
from edp.pipeline import run

app = FastAPI(title="Electrical Drawing Interpretation Pipeline")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    cfg = Config.load()

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result, timing = run(tmp_path, cfg)
        json_dict = to_json_dict(result)

        with tempfile.TemporaryDirectory() as out_dir:
            graph_paths = export_all(result, out_dir)
            graph_png_b64 = base64.b64encode(Path(graph_paths["png"]).read_bytes()).decode()

        return JSONResponse(
            {
                "json": json_dict,
                "graph_png": f"data:image/png;base64,{graph_png_b64}",
                "timing": timing,
            }
        )
    finally:
        tmp_path.unlink(missing_ok=True)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
