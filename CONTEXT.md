# find-photo Domain Context

## Glossary

### Race Photo (대회 사진)
A photo taken during a running event (marathon or trail running). Stored in a shared Google Drive folder. May contain hundreds of runners, many wearing hats and sunglasses.

### Bib Number (배번호)
The numbered identifier worn on a runner's chest during a race. Primary search key. Changes per event. Entered by the user at search time.

### Reference Photo (기준 사진)
A photo of the user's face uploaded at search time. Used to generate a face embedding for matching against indexed race photos. Uploaded fresh each search session.

### Drive Folder (드라이브 폴더)
A shared Google Drive folder URL containing race photos for a specific event. Entered by the user at search time. One folder = one event's photos.

### Index (인덱스)
Pre-computed data extracted from all photos in a Drive Folder:
- Face embeddings (vector per detected face)
- OCR-extracted bib numbers

Stored locally in a database keyed by Drive Folder ID. Enables fast search without re-downloading photos.

### Match (매칭)
A Race Photo identified as containing the user. Found by comparing the user's Bib Number and/or Reference Photo against the Index.

### Match Type (매칭 유형)
How a Match was found:
- `bib` — bib number matched
- `face` — face embedding matched
- `both` — both matched

Displayed as a badge on each result. Used to help the user judge confidence.

### Indexing (인덱싱)
The one-time process of downloading all photos from a Drive Folder, running face detection and OCR, and storing results in the Index. Must be run once per Drive Folder before search is possible.
