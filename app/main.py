import json
import threading
import asyncio

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.bib_extractor import BibExtractor
from app.drive_connector import DriveConnector
from app.face_detector import FaceDetector
from app.index_store import IndexStore
from app import indexer as indexer_module
from app import search_engine

app = FastAPI(title="find-photo")

connector = DriveConnector()
store = IndexStore()
bib_extractor = BibExtractor()
face_detector = FaceDetector()


@app.on_event("startup")
def startup():
    from pathlib import Path
    if Path("data/token.json").exists():
        try:
            connector.authenticate()
        except Exception:
            pass


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.get("/api/auth/status")
def auth_status():
    return {"authenticated": connector.is_authenticated()}


@app.post("/api/auth/login")
def login():
    try:
        connector.authenticate()
        return {"authenticated": True}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="credentials.json not found")


# ── Drive folder ──────────────────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    folder_url: str


@app.post("/api/connect")
def connect(req: ConnectRequest):
    if not connector.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        folder_id = DriveConnector.parse_folder_id(req.folder_url)
        photos = connector.list_photos(req.folder_url)
        status = store.get_folder_status(folder_id)
        return {
            "folder_id": folder_id,
            "photo_count": len(photos),
            "index_status": status,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Indexing ──────────────────────────────────────────────────────────────────

@app.get("/api/index/{folder_id}/status")
def index_status(folder_id: str):
    status = store.get_folder_status(folder_id)
    if not status:
        return {"status": "not_indexed"}
    return status


@app.post("/api/index")
async def start_index(req: ConnectRequest):
    if not connector.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated")

    async def generate():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_in_thread():
            try:
                for progress in indexer_module.run(
                    req.folder_url, connector, store, bib_extractor, face_detector
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, progress)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"stage": "error", "message": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/index/{folder_id}")
def clear_index(folder_id: str):
    store.clear_folder(folder_id)
    return {"cleared": True}


# ── Search ────────────────────────────────────────────────────────────────────

@app.post("/api/search")
async def search(
    folder_url: str = Form(...),
    bib_number: str = Form(...),
    reference_photo: UploadFile | None = File(default=None),
):
    folder_id = DriveConnector.parse_folder_id(folder_url)
    if not store.is_indexed(folder_id):
        raise HTTPException(status_code=400, detail="폴더가 인덱싱되지 않았습니다")

    ref_embedding: np.ndarray | None = None
    if reference_photo:
        img_bytes = await reference_photo.read()
        embeddings = face_detector.detect_faces(img_bytes)
        if embeddings:
            ref_embedding = embeddings[0]

    indexed_photos = store.get_all_photos(folder_id)
    results = search_engine.search(indexed_photos, ref_embedding, bib_number)

    return {
        "results": [
            {
                "photo_id": r.photo_id,
                "photo_name": r.photo_name,
                "drive_url": r.drive_url,
                "match_type": r.match_type,
                "face_score": r.face_score,
                "thumbnail_url": f"https://drive.google.com/thumbnail?id={r.photo_id}&sz=w400",
            }
            for r in results
        ],
        "total": len(results),
    }


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
