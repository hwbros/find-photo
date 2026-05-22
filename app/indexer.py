import io
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from PIL import Image

from app.bib_extractor import BibExtractor
from app.drive_connector import DriveConnector
from app.face_detector import FaceDetector
from app.index_store import IndexStore

MAX_IMAGE_PX = 800
DOWNLOAD_WORKERS = 4


def _resize(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if max(img.size) > MAX_IMAGE_PX:
        img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run(
    folder_url: str,
    connector: DriveConnector,
    store: IndexStore,
    bib_extractor: BibExtractor,
    face_detector: FaceDetector | None = None,
) -> Generator[dict, None, None]:
    folder_id = DriveConnector.parse_folder_id(folder_url)

    yield {"stage": "scanning", "message": "폴더 스캔 중..."}
    photos = connector.list_photos(folder_url)
    total = len(photos)

    indexed_ids = store.get_indexed_photo_ids(folder_id)
    to_process = [p for p in photos if p.id not in indexed_ids]
    already_done = len(indexed_ids)

    store.init_folder(folder_id, folder_url, total)
    yield {"stage": "indexing", "current": already_done, "total": total}

    if not to_process:
        store.mark_complete(folder_id)
        yield {"stage": "done", "total": total}
        return

    # Pipeline: DOWNLOAD_WORKERS threads download + resize,
    # main thread does OCR + save (EasyOCR is not thread-safe)
    download_q: queue.Queue = queue.Queue()

    def download_one(photo):
        try:
            raw = connector.download_photo(photo.id)
            download_q.put((photo, _resize(raw)))
        except Exception:
            download_q.put((photo, None))

    processed = already_done
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        for p in to_process:
            executor.submit(download_one, p)

        for _ in to_process:
            photo, img_bytes = download_q.get()
            try:
                bibs = bib_extractor.extract_bibs(img_bytes) if img_bytes else []
            except Exception:
                bibs = []
            try:
                embeddings = face_detector.detect_faces(img_bytes) if (face_detector and img_bytes) else []
            except Exception:
                embeddings = []
            store.save_photo(folder_id, photo.id, photo.name, photo.web_view_link, bibs, embeddings)
            processed += 1
            yield {
                "stage": "indexing",
                "current": processed,
                "total": total,
                "name": photo.name,
            }

    store.mark_complete(folder_id)
    yield {"stage": "done", "total": total}
