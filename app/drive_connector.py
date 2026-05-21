import re
import io
from dataclasses import dataclass
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/heic",
}


@dataclass
class PhotoMeta:
    id: str
    name: str
    thumbnail_url: str | None
    web_view_link: str


class DriveConnector:
    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path: str = "data/token.json",
    ):
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._service = None

    def authenticate(self) -> None:
        creds = None
        token_file = Path(self._token_path)
        token_file.parent.mkdir(parents=True, exist_ok=True)

        if token_file.exists():
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self._credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_file.write_text(creds.to_json())

        self._service = build("drive", "v3", credentials=creds)

    def is_authenticated(self) -> bool:
        return self._service is not None

    @staticmethod
    def parse_folder_id(folder_url: str) -> str:
        match = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_url)
        if not match:
            raise ValueError(f"Cannot parse folder ID from URL: {folder_url}")
        return match.group(1)

    def list_photos(self, folder_url: str) -> list[PhotoMeta]:
        if not self._service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        folder_id = self.parse_folder_id(folder_url)
        return self._list_photos_recursive(folder_id)

    def _list_photos_recursive(self, folder_id: str) -> list[PhotoMeta]:
        photos = []
        page_token = None

        while True:
            resp = (
                self._service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, thumbnailLink, webViewLink)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )

            for f in resp.get("files", []):
                mime = f.get("mimeType", "")
                if mime == "application/vnd.google-apps.folder":
                    photos.extend(self._list_photos_recursive(f["id"]))
                elif mime in IMAGE_MIME_TYPES:
                    photos.append(
                        PhotoMeta(
                            id=f["id"],
                            name=f["name"],
                            thumbnail_url=f.get("thumbnailLink"),
                            web_view_link=f.get("webViewLink", ""),
                        )
                    )

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return photos

    def download_photo(self, photo_id: str) -> bytes:
        if not self._service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        request = self._service.files().get_media(fileId=photo_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()
