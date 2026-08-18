"""InVision web service.

Wraps the localiser and the navigation graph behind a small JSON API and serves
the map interface.

Run:
    python api.py
    # or: uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.mall_graph import (
    NEARBY_LIMIT,
    NEARBY_MAX_RADIUS_M,
    NEARBY_RADIUS_M,
    MallGraph,
)

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The graph is cheap; the localiser pulls CLIP into memory, so it is loaded
    # once at startup rather than per request.
    state["graph"] = MallGraph.from_derived()
    from src.localizer import Localizer

    state["localizer"] = Localizer()
    yield
    state.clear()


app = FastAPI(title="InVision", lifespan=lifespan)


class RouteRequest(BaseModel):
    origin_id: str
    destination_id: str


class TourRequest(BaseModel):
    origin_id: str
    stops: list[str]


@app.get("/api/stores")
def list_stores():
    """Every unit on the floor directory. All of them are routable."""
    graph: MallGraph = state["graph"]
    units = sorted(
        graph.units.values(),
        key=lambda u: (u["floor"], u["row"], u["index"]),
    )
    return {
        "stores": [
            {
                "store_id": u["unit_id"],
                "name": u["name"],
                "floor": u["floor"],
                "floor_name": u["floor_name"],
                "row": u["row"],
                "index": u["index"],
                "has_images": u["image_count"] > 0,
                "routable": True,
            }
            for u in units
        ]
    }


@app.get("/api/layout")
def layout():
    return state["graph"].layout()


async def _localise(photo: UploadFile) -> dict:
    """Decode an upload and run it through the localiser.

    Shared by the opening "where am I" question and the progress checks that
    follow it, which ask the same thing of the same model.
    """
    raw = await photo.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode that image")

    # The localiser and OCR engine both read from a path, so the upload is
    # staged to a temporary file rather than kept in memory.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        temp_path = Path(handle.name)
        image.save(handle, "JPEG", quality=92)

    try:
        predictions = state["localizer"].locate(temp_path, top_k=5)
    finally:
        temp_path.unlink(missing_ok=True)

    if not predictions:
        raise HTTPException(422, "no match found")

    top, second = predictions[0], (predictions[1] if len(predictions) > 1 else None)
    margin = top.score - second.score if second else 1.0
    confident = top.decided_by == "signage" or (
        top.decided_by == "fusion" and margin >= 0.03
    )

    enriched = []
    for p in predictions:
        row = asdict(p)
        row["unit_id"] = p.unit_id
        enriched.append(row)

    return {
        "predictions": enriched,
        "confident": confident,
        "margin": round(margin, 4),
        "decided_by": top.decided_by,
    }


@app.post("/api/locate")
async def locate(photo: UploadFile = File(...)):
    return await _localise(photo)


@app.post("/api/progress")
async def progress(
    photo: UploadFile = File(...),
    origin_id: str = Form(...),
    destination_id: str = Form(...),
):
    """Mid-walk check: a second photo, judged against the route already given.

    Answers whether the shopper is still on the line, and hands back a route
    from wherever they actually are - the same one if they are on track, a
    fresh one if they wandered.
    """
    found = await _localise(photo)
    here = found["predictions"][0]["unit_id"]

    graph: MallGraph = state["graph"]
    try:
        result = graph.progress(origin_id, here, destination_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    route = asdict(result.route)
    route["origin_id"] = here
    route["destination_id"] = destination_id

    return {
        "state": result.state,
        "current": found["predictions"][0],
        "confident": found["confident"],
        "alternatives": found["predictions"][1:4],
        "deviation_m": result.deviation_m,
        "remaining_m": result.remaining_m,
        "original_m": result.original_m,
        "done_fraction": result.done_fraction,
        "minutes_left": result.minutes_left,
        "route": route,
    }


@app.post("/api/route")
def route(request: RouteRequest):
    graph: MallGraph = state["graph"]
    try:
        result = graph.route(request.origin_id, request.destination_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    return {
        "steps": [asdict(s) for s in result.steps],
        "total_distance_m": result.total_distance_m,
        "floors_traversed": result.floors_traversed,
        "caveats": result.caveats,
        "map_legs": result.map_legs,
        "origin_id": request.origin_id,
        "destination_id": request.destination_id,
    }


@app.get("/api/nearby")
def nearby(
    unit_id: str,
    radius_m: float = NEARBY_RADIUS_M,
    limit: int = NEARBY_LIMIT,
):
    """What a shopper can reach on foot from where they are standing.

    The radius and the count are clamped rather than rejected: this is asked
    on every recognised sighting, and a silly query should still answer.
    """
    graph: MallGraph = state["graph"]
    radius = min(max(float(radius_m), 0.0), NEARBY_MAX_RADIUS_M)
    count = min(max(int(limit), 1), 50)

    try:
        found = graph.nearby(unit_id, radius_m=radius, limit=count)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    return {
        "origin_id": graph.resolve(unit_id),
        "radius_m": radius,
        "nearby": [asdict(n) for n in found],
    }


@app.post("/api/tour")
def tour(request: TourRequest):
    """A shopping list, ordered so the walk is as short as it can be."""
    graph: MallGraph = state["graph"]
    try:
        result = graph.tour(request.origin_id, request.stops)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    legs = []
    for start, end, leg in zip(result.order, result.order[1:], result.legs):
        row = asdict(leg)
        row["origin_id"] = start
        row["destination_id"] = end
        row["destination_name"] = graph.units[end]["name"]
        legs.append(row)

    return {
        "order": [
            {
                "unit_id": unit_id,
                "name": graph.units[unit_id]["name"],
                "floor": graph.units[unit_id]["floor"],
                "floor_name": graph.units[unit_id]["floor_name"],
            }
            for unit_id in result.order
        ],
        "legs": legs,
        "total_distance_m": result.total_distance_m,
        "minutes": result.minutes,
        "naive_distance_m": result.naive_distance_m,
        "saved_m": result.saved_m,
    }


# ---------------------------------------------------------------- review pass

REVIEW_FILE = Path(__file__).parent / "data" / "label_review.json"
THUMB_DIR = Path(__file__).parent / "data" / "derived" / "thumbs"
THUMB_MAX = 360


class ReviewSave(BaseModel):
    assignments: dict[str, str | None]
    confirmed: list[str] = []


def _load_review() -> dict:
    if REVIEW_FILE.exists():
        return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    return {"assignments": {}, "confirmed": []}


@app.get("/api/thumb")
def thumb(path: str):
    """Serve a cached thumbnail of a captured photograph."""
    from src import config as cfg

    source = (cfg.IMAGES_DIR / path).resolve()
    # Keep the endpoint from reaching outside the capture folder.
    if not str(source).startswith(str(cfg.IMAGES_DIR.resolve())) or not source.is_file():
        raise HTTPException(404, "no such image")

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cached = THUMB_DIR / (hashlib.sha1(path.encode()).hexdigest() + ".jpg")
    if not cached.exists():
        with Image.open(source) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
            img.save(cached, "JPEG", quality=82)
    return FileResponse(cached, media_type="image/jpeg")


@app.get("/api/review")
def review_data():
    """Every captured frame in walk order, with its current unit assignment."""
    from src import config as cfg

    graph: MallGraph = state["graph"]
    derived = cfg.DERIVED_DIR
    records = json.loads((derived / "all_index.json").read_text(encoding="utf-8"))
    stores = json.loads((derived / "stores.json").read_text(encoding="utf-8"))
    ocr = json.loads((derived / "ocr_raw.json").read_text(encoding="utf-8"))
    review = _load_review()

    unit_of_image: dict[str, str] = {}
    for store in stores:
        unit_id = graph.unit_for_store(store["store_id"])
        if unit_id:
            for image in store["images"]:
                unit_of_image[image] = unit_id

    corridors: dict[tuple, dict] = {}
    for rec in sorted(records, key=lambda r: (r["floor"], r["side"], r["order"])):
        key = (rec["floor"], rec["side"])
        if key not in corridors:
            row = next(
                (graph.units[u]["row"] for u in unit_of_image.values()
                 if graph.units[u]["floor"] == rec["floor"]),
                None,
            )
            corridors[key] = {
                "floor": rec["floor"],
                "floor_name": rec["floor_name"],
                "side": rec["side"],
                "frames": [],
            }
        detected = [
            t["text"] for t in ocr.get(rec["rel_path"], {}).get("tokens", [])[:4]
        ]
        auto = unit_of_image.get(rec["rel_path"])
        corridors[key]["frames"].append({
            "path": rec["rel_path"],
            "auto_unit": auto,
            "unit": review["assignments"].get(rec["rel_path"], auto),
            "confirmed": rec["rel_path"] in set(review["confirmed"]),
            "ocr": detected,
        })

    units_by_floor: dict[str, list] = {}
    for unit in graph.units.values():
        units_by_floor.setdefault(str(unit["floor"]), []).append({
            "unit_id": unit["unit_id"], "name": unit["name"],
            "row": unit["row"], "index": unit["index"],
        })
    for items in units_by_floor.values():
        items.sort(key=lambda u: (u["row"], u["index"]))

    return {
        "corridors": list(corridors.values()),
        "units_by_floor": units_by_floor,
        "total_frames": len(records),
    }


@app.post("/api/review")
def review_save(payload: ReviewSave):
    review = _load_review()
    review["assignments"].update(payload.assignments)
    review["confirmed"] = sorted(set(review["confirmed"]) | set(payload.confirmed))
    review["updated"] = datetime.now().isoformat(timespec="seconds")
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "saved": len(payload.assignments),
        "confirmed_total": len(review["confirmed"]),
        "file": str(REVIEW_FILE),
    }


@app.get("/review")
def review_page():
    page = WEB_DIR / "review.html"
    if not page.exists():
        return JSONResponse({"error": "web/review.html missing"}, status_code=500)
    return FileResponse(page)


NO_CACHE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


@app.get("/")
def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"error": "web/index.html missing"}, status_code=500)
    return FileResponse(page, headers=NO_CACHE)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
