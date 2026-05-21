# PRD: Race Photo Finder

## Problem Statement

Runners participating in marathons and trail running events end up with thousands of official race photos from each event. Finding photos that actually contain themselves requires manually browsing through all of them — a tedious process that often takes hours. Photos are typically shared in a Google Drive folder by event organizers.

The challenge is compounded by the fact that runners wear hats and sunglasses, making face recognition harder, and that each event assigns a different bib number.

## Solution

A locally-run web application that lets a runner find their own photos from a shared Google Drive folder by combining two identification methods:

1. **Bib number OCR** — recognizes the runner's race bib number in photos
2. **Face recognition** — matches the runner's face against a reference photo they upload

The app pre-indexes a Drive folder once (face embeddings + bib numbers), then returns search results in seconds. Each matched photo is displayed as a thumbnail with a badge indicating how it was matched (`bib`, `face`, or `both`), and links back to the original file in Google Drive.

## User Stories

1. As a runner, I want to paste a Google Drive folder URL so that I can search photos from a specific race event.
2. As a runner, I want to enter my bib number so that the app can find photos where my number is visible.
3. As a runner, I want to upload a reference photo of my face so that the app can recognize me even when my bib is obscured.
4. As a runner, I want both bib number and face recognition to run together (OR condition) so that I don't miss photos where only one method works.
5. As a runner, I want to see a progress indicator while a Drive folder is being indexed so that I know the system is working.
6. As a runner, I want indexing to happen only once per folder so that subsequent searches are fast.
7. As a runner, I want search results displayed as a thumbnail grid so that I can quickly scan which photos are mine.
8. As a runner, I want each thumbnail to show a `bib`, `face`, or `both` badge so that I can judge how confident the match is.
9. As a runner, I want each thumbnail to link to the original photo in Google Drive so that I can view or download full resolution.
10. As a runner, I want to upload a fresh reference photo each search session so that I can use different photos without managing a profile.
11. As a runner, I want to use the app for different race events by entering a different Drive URL each time so that I don't need to reconfigure anything between events.
12. As a runner, I want the app to handle hats and sunglasses in race photos so that face recognition still works under typical race conditions.
13. As a runner, I want partial bib number matches filtered out so that I don't see photos of other runners whose bib contains my number as a substring.
14. As a runner, I want to see a "no results found" message when neither method finds a match so that I know the search completed rather than failed silently.
15. As a runner, I want to re-index a folder if I know new photos have been added so that the index stays current.
16. As a runner, I want the app to run locally on my PC so that my face data and Drive credentials never leave my machine.

## Implementation Decisions

### Module breakdown

Six modules, four of which are deep (high complexity behind a simple interface):

**1. DriveConnector** *(deep)*
Encapsulates all Google Drive API interaction: OAuth 2.0 flow, folder URL parsing, photo listing (with pagination), and photo downloading. Caller only sees:
- `list_photos(folder_url) -> list[PhotoMeta]`
- `download_photo(photo_id) -> bytes`

**2. FaceDetector** *(deep)*
Handles face detection and 128-dimension embedding extraction from raw image bytes. Uses `face_recognition` (dlib-based). Returns one embedding per detected face.
- `detect_faces(image_bytes) -> list[FaceEmbedding]`

**3. BibExtractor** *(deep)*
Runs OCR on image bytes and returns candidate bib numbers found (as strings). Uses `EasyOCR`. Filters OCR output to numeric-only tokens in plausible bib-number regions.
- `extract_bibs(image_bytes) -> list[str]`

**4. IndexStore** *(deep)*
SQLite-backed store. Persists per-photo indexed data (face embeddings serialized as numpy arrays, bib number strings) keyed by Drive folder ID and photo ID. Exposes:
- `is_indexed(folder_id) -> bool`
- `save_photo(folder_id, photo_id, photo_url, embeddings, bibs)`
- `get_all(folder_id) -> list[IndexedPhoto]`
- `clear_folder(folder_id)`

**5. SearchEngine** *(deep)*
Given an IndexStore result set, a reference photo, and a bib number, computes cosine similarity between the reference face embedding and all stored embeddings, and performs exact string match on bib numbers. Returns ranked MatchResult list with match type.
- `search(indexed_photos, reference_bytes, bib_number) -> list[MatchResult]`
- `MatchResult` carries: `photo_id`, `photo_url`, `match_type` (`bib | face | both`), `face_score`

**6. Indexer** *(shallow orchestrator)*
Coordinates DriveConnector → FaceDetector + BibExtractor → IndexStore. Streams progress events (photo count, current photo name) to the caller for UI feedback.

**7. WebApp** *(FastAPI + static HTML/JS)*
Thin HTTP layer. Endpoints:
- `POST /index` — start indexing a Drive folder (streams progress via SSE)
- `GET /index/{folder_id}/status` — check if a folder is indexed
- `POST /search` — multipart form: `folder_url`, `bib_number`, `reference_photo`
- `DELETE /index/{folder_id}` — clear index for re-indexing

### Key architectural decisions

- **OR condition for matching**: a photo is included in results if bib OR face matches (not both required). Match type badge reflects which fired.
- **Face similarity threshold**: cosine distance ≤ 0.6 is treated as a match (standard for `face_recognition` library). Tunable via config.
- **Bib matching**: exact string match after stripping leading zeros. Substring matches excluded.
- **Index persistence**: SQLite file at `./data/index.db`. Face embeddings stored as binary-serialized numpy float64 arrays.
- **Google Drive auth**: OAuth 2.0 with `credentials.json` from Google Cloud Console. Token cached at `./data/token.json`.
- **Deployment**: local-only. No authentication layer, no multi-user support.

## Testing Decisions

**What makes a good test here:** test observable outputs given inputs, not internal steps. For example: given an image containing a bib numbered "1234", `BibExtractor.extract_bibs()` returns `["1234"]` — don't test which OCR model was called.

**Modules to test:**

| Module | Test type | What to test |
|---|---|---|
| `BibExtractor` | Unit | Returns correct bib numbers from sample race images; rejects non-numeric noise; ignores substrings |
| `FaceDetector` | Unit | Returns embeddings for photos with faces; returns empty list for photos with no face; handles hat/glasses |
| `IndexStore` | Integration | Round-trip save and retrieve; `is_indexed` reflects state; `clear_folder` removes data |
| `SearchEngine` | Unit | `face` match fires when embedding is close enough; `bib` match fires on exact number; `both` fires when both match; no match when neither matches; face threshold boundary conditions |

`DriveConnector` and `Indexer` are excluded from unit tests — their logic is mostly I/O orchestration, best covered by manual integration testing with a real Drive folder.

## Out of Scope

- Multi-user accounts or authentication
- Cloud deployment (app is local-only)
- Video files or non-image formats
- Batch download of matched photos (results link to Drive; download is manual)
- Support for Drive folders requiring per-user Google account login (public shared folders only)
- Mobile-responsive UI
- Automatic re-indexing when Drive folder changes

## Further Notes

- Google Cloud project setup (enabling Drive API, creating OAuth credentials) is a prerequisite. A setup guide will be provided as part of the implementation.
- The `face_recognition` library requires `cmake` and `dlib` to install on Windows — installation steps will be documented.
- For very large events (5,000+ photos), indexing may take 2–4 hours. A resume-on-interrupt feature (skip already-indexed photos) is worth adding to the Indexer but is not in scope for v1.
- Face recognition accuracy under heavy hat/sunglasses varies. Bib number matching is the more reliable primary signal; face matching improves recall.
