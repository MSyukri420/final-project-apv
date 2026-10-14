from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.apv_engine import APVEngine
from src.coach_service import ALL_SHAPES, CoachService

WEB_DIR = Path(__file__).parent / "web"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "7860"))


def load_config(path: Path = Path("config.yaml")) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


SERVICE = CoachService(APVEngine(load_config()))
# the engine isn't safe for parallel calls, so rate one setup at a time
ENGINE_LOCK = threading.Lock()


class SetupIn(BaseModel):
    shape: str
    delivery: str
    zone: str
    players: Dict[str, int]


app = FastAPI(title="Corner Chance Planner")


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    return SERVICE.meta()


@app.post("/api/evaluate")
def evaluate(setup: SetupIn) -> Dict[str, Any]:
    try:
        with ENGINE_LOCK:
            return SERVICE.evaluate(setup.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/playbook")
def playbook(
    shape: str = ALL_SHAPES,
    min_used: int = 10,
    limit: int = 9,
    sort: str = "chance",
    include_other: bool = False,
) -> Dict[str, Any]:
    return SERVICE.playbook(
        shape=shape,
        min_used=max(min_used, 1),
        limit=min(max(limit, 1), 60),
        sort=sort,
        include_other=include_other,
    )


@app.get("/api/insights")
def insights() -> Dict[str, Any]:
    return SERVICE.insights()


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> HTMLResponse:
    # add each file's time to its link, so browsers never run an old app.js after an update
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "styles.css"):
        version = int((WEB_DIR / name).stat().st_mtime)
        html = html.replace(f"/static/{name}", f"/static/{name}?v={version}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    print(f"\nCorner Chance Planner running at http://{HOST}:{PORT}  (Ctrl+C to stop)\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
