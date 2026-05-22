import io
import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from PIL import Image

from app.bib_extractor import BibExtractor
from app.drive_connector import DriveConnector
from app.face_detector import FaceDetector
from app.index_store import IndexStore

log = logging.getLogger(__name__)

MAX_IMAGE_PX = 1600
DOWNLOAD_WORKERS = 4
SUBMIT_BATCH = 50  # submit tasks in batches to avoid memory pressure


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

    log.info("indexer: scanning folder %s", folder_id)
    yield {"stage": "scanning", "message": "폴더 스캔 중..."}
    photos = connector.list_photos(folder_url)
    total = len(photos)
    log.info("indexer: found %d photos", total)

    indexed_ids = store.get_indexed_photo_ids(folder_id)
    to_process = [p for p in photos if p.id not in indexed_ids]
    already_done = len(indexed_ids)
    log.info("indexer: %d already indexed, %d to process", already_done, len(to_process))

    store.init_folder(folder_id, folder_url, total)
    yield {"stage": "indexing", "current": already_done, "total": total}

    if not to_process:
        store.mark_complete(folder_id)
        yield {"stage": "done", "total": total}
        return

    processed = already_done

    # Submit tasks in batches so we don't queue 17k download futures at once
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        offset = 0
        while offset < len(to_process):
            batch = to_process[offset : offset + SUBMIT_BATCH]
            offset += SUBMIT_BATCH

            download_q: queue.Queue = queue.Queue()

            def download_one(photo, _q=download_q):
                try:
                    log.debug("downloading %s", photo.name)
                    raw = connector.download_photo_threadsafe(photo.id)
                    _q.put((photo, _resize(raw)))
                except Exception as exc:
                    log.warning("download failed for %s: %s", photo.name, exc)
                    _q.put((photo, None))

            for p in batch:
                executor.submit(download_one, p)

            for _ in batch:
                photo, img_bytes = download_q.get()
                bibs: list[str] = []
                embeddings = []
                if img_bytes:
                    try:
                        bibs = bib_extractor.extract_bibs(img_bytes)
                        log.debug("%s → bibs %s", photo.name, bibs)
                    except Exception as exc:
                        log.warning("OCR failed for %s: %s", photo.name, exc)
                    if face_detector is not None:
                        try:
                            embeddings = face_detector.detect_faces(img_bytes)
                            log.debug("%s → %d face(s)", photo.name, len(embeddings))
                        except Exception as exc:
                            log.warning("face detection failed for %s: %s", photo.name, exc)
                store.save_photo(
                    folder_id, photo.id, photo.name, photo.web_view_link, bibs, embeddings
                )
                processed += 1
                yield {
                    "stage": "indexing",
                    "current": processed,
                    "total": total,
                    "name": photo.name,
                }

    store.mark_complete(folder_id)
    log.info("indexer: done (%d photos)", total)
    yield {"stage": "done", "total": total}
