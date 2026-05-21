from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.drive_connector import DriveConnector, PhotoMeta

app = FastAPI(title="find-photo")

connector = DriveConnector()


@app.on_event("startup")
def startup():
    from pathlib import Path
    if Path("data/token.json").exists():
        try:
            connector.authenticate()
        except Exception:
            pass


class ConnectRequest(BaseModel):
    folder_url: str


class ConnectResponse(BaseModel):
    folder_id: str
    photo_count: int
    photos: list[PhotoMeta]


@app.get("/api/auth/status")
def auth_status():
    return {"authenticated": connector.is_authenticated()}


@app.post("/api/auth/login")
def login():
    try:
        connector.authenticate()
        return {"authenticated": True}
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="credentials.json not found. See docs/setup-google-cloud.md",
        )



@app.post("/api/connect", response_model=ConnectResponse)
def connect(req: ConnectRequest):
    if not connector.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        folder_id = DriveConnector.parse_folder_id(req.folder_url)
        photos = connector.list_photos(req.folder_url)
        return ConnectResponse(
            folder_id=folder_id,
            photo_count=len(photos),
            photos=photos,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
