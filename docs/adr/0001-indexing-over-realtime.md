# ADR 0001: Pre-index photos instead of real-time processing

## Status
Accepted

## Context
Searching thousands of race photos requires downloading each image and running face detection + OCR. At 1–3 seconds per photo, processing 1,000 photos in real-time would take 15–30 minutes per search.

## Decision
Pre-compute and store face embeddings and bib numbers for all photos in a Drive Folder on first scan (Indexing). Subsequent searches query the local Index only.

## Consequences
- First scan of a new Drive Folder takes time (acceptable — runs once per event)
- Search results return in seconds after indexing
- Index must be rebuilt if Drive Folder contents change significantly
- Local storage required for the Index database
