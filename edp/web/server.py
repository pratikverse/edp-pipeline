"""Thin FastAPI wrapper around edp.pipeline.run(). Holds no pipeline logic
of its own — see docs/07_project_layout.md ("Demo frontend")."""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from edp.config import Config
from edp.emit import export_all, to_json_dict
from edp.pipeline import run

app = FastAPI(title="Electrical Drawing Interpretation Pipeline")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _render_detection_overlay(image_path: Path, symbols: list[dict]) -> bytes:
    """Draws each detected symbol's box + id + type on the source image —
    the same visual audit view used throughout docs/05's accuracy work,
    surfaced in the demo instead of living only in ad-hoc debug scripts."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    for symbol in symbols:
        x0, y0, x1, y1 = symbol["coordinates"]
        color = (0, 140, 255) if symbol["type"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        label = f"{symbol['id']} {symbol['type']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(img, label, (x0 + 2, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    cfg = Config.load()
    original_bytes = await file.read()

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(original_bytes)
        tmp_path = Path(tmp.name)

    try:
        result, timing = run(tmp_path, cfg)
        json_dict = to_json_dict(result)

        overlay_bytes = _render_detection_overlay(tmp_path, json_dict["symbols"])

        with tempfile.TemporaryDirectory() as out_dir:
            graph_paths = export_all(result, out_dir)
            graph_png_b64 = base64.b64encode(Path(graph_paths["png"]).read_bytes()).decode()

        original_b64 = base64.b64encode(original_bytes).decode()
        overlay_b64 = base64.b64encode(overlay_bytes).decode()
        original_mime = "image/png" if suffix.lower() == ".png" else "image/jpeg"

        return JSONResponse(
            {
                "json": json_dict,
                "original_png": f"data:{original_mime};base64,{original_b64}",
                "overlay_png": f"data:image/png;base64,{overlay_b64}",
                "graph_png": f"data:image/png;base64,{graph_png_b64}",
                "timing": timing,
            }
        )
    finally:
        tmp_path.unlink(missing_ok=True)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
