import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class IndexStore:
    def __init__(self, db_path: str = "data/index.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS folders (
                    folder_id   TEXT PRIMARY KEY,
                    folder_url  TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'indexing',
                    total       INTEGER NOT NULL DEFAULT 0,
                    indexed     INTEGER NOT NULL DEFAULT 0,
                    started_at  TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS photos (
                    folder_id   TEXT NOT NULL,
                    photo_id    TEXT NOT NULL,
                    photo_name  TEXT NOT NULL,
                    drive_url   TEXT NOT NULL,
                    PRIMARY KEY (folder_id, photo_id)
                );
                CREATE TABLE IF NOT EXISTS photo_bibs (
                    folder_id   TEXT NOT NULL,
                    photo_id    TEXT NOT NULL,
                    bib_number  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_photo_bibs
                    ON photo_bibs(folder_id, bib_number);
            """)

    def is_indexed(self, folder_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM folders WHERE folder_id = ?", (folder_id,)
            ).fetchone()
            return row is not None and row["status"] == "complete"

    def is_photo_indexed(self, folder_id: str, photo_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM photos WHERE folder_id = ? AND photo_id = ?",
                (folder_id, photo_id),
            ).fetchone()
            return row is not None

    def init_folder(self, folder_id: str, folder_url: str, total: int):
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO folders (folder_id, folder_url, status, total, indexed, started_at)
                   VALUES (?, ?, 'indexing', ?, 0, ?)
                   ON CONFLICT(folder_id) DO UPDATE SET
                       status='indexing', total=excluded.total, indexed=0, started_at=excluded.started_at, completed_at=NULL""",
                (folder_id, folder_url, total, now),
            )

    def save_photo(
        self,
        folder_id: str,
        photo_id: str,
        photo_name: str,
        drive_url: str,
        bibs: list[str],
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO photos (folder_id, photo_id, photo_name, drive_url)
                   VALUES (?, ?, ?, ?)""",
                (folder_id, photo_id, photo_name, drive_url),
            )
            conn.execute(
                "DELETE FROM photo_bibs WHERE folder_id = ? AND photo_id = ?",
                (folder_id, photo_id),
            )
            for bib in bibs:
                conn.execute(
                    "INSERT INTO photo_bibs (folder_id, photo_id, bib_number) VALUES (?, ?, ?)",
                    (folder_id, photo_id, bib),
                )
            conn.execute(
                "UPDATE folders SET indexed = indexed + 1 WHERE folder_id = ?",
                (folder_id,),
            )

    def mark_complete(self, folder_id: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE folders SET status='complete', completed_at=? WHERE folder_id=?",
                (now, folder_id),
            )

    def get_folder_status(self, folder_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM folders WHERE folder_id = ?", (folder_id,)
            ).fetchone()
            return dict(row) if row else None

    def search_by_bib(self, folder_id: str, bib_number: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT p.photo_id, p.photo_name, p.drive_url
                   FROM photos p
                   JOIN photo_bibs pb ON p.folder_id = pb.folder_id AND p.photo_id = pb.photo_id
                   WHERE p.folder_id = ? AND pb.bib_number = ?""",
                (folder_id, bib_number),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_folder(self, folder_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM photo_bibs WHERE folder_id = ?", (folder_id,))
            conn.execute("DELETE FROM photos WHERE folder_id = ?", (folder_id,))
            conn.execute("DELETE FROM folders WHERE folder_id = ?", (folder_id,))
