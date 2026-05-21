from typing import Generator

from app.bib_extractor import BibExtractor
from app.drive_connector import DriveConnector
from app.index_store import IndexStore


def run(
    folder_url: str,
    connector: DriveConnector,
    store: IndexStore,
    bib_extractor: BibExtractor,
) -> Generator[dict, None, None]:
    folder_id = DriveConnector.parse_folder_id(folder_url)

    yield {"stage": "scanning", "message": "폴더 스캔 중..."}
    photos = connector.list_photos(folder_url)
    total = len(photos)

    store.init_folder(folder_id, folder_url, total)
    yield {"stage": "indexing", "current": 0, "total": total}

    for i, photo in enumerate(photos):
        if store.is_photo_indexed(folder_id, photo.id):
            yield {"stage": "indexing", "current": i + 1, "total": total, "name": photo.name}
            continue
        try:
            image_bytes = connector.download_photo(photo.id)
            bibs = bib_extractor.extract_bibs(image_bytes)
        except Exception:
            bibs = []
        store.save_photo(folder_id, photo.id, photo.name, photo.web_view_link, bibs)
        yield {"stage": "indexing", "current": i + 1, "total": total, "name": photo.name}

    store.mark_complete(folder_id)
    yield {"stage": "done", "total": total}
