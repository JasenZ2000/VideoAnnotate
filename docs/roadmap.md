# Roadmap And Known Limitations

## Required For Production Workflow

- Make package generation, reviewed-result upload and final export fully segment-aware.
- Add segment assignment/progress controls and final multi-segment merge validation.
- Persist remote job state so service restarts do not lose polling metadata.
- Add authentication, authorization and audit identities to the platform.
- Add cancellation, retry and queue visibility for long GPU jobs.
- Package Annotator as a tested Windows distribution with upgrade/version reporting.

## Data Quality

- Define strict behavior for unknown LocateAnything class labels instead of silently using the fallback class ID.
- Add automatic frame-offset validation for uploaded YOLO archives.
- Add dataset-level checks for missing frames, invalid boxes, class drift and duplicate trajectories.

## Operations

- Add retention/cleanup policy for GPU caches and old task artifacts.
- Add structured logs and basic service metrics.
- Add backup/restore drills and a task schema migration mechanism.

These items are explicit so operators can distinguish available MVP behavior from intended platform behavior.
