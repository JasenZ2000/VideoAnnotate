# Data And Directory Contract

## Platform Task Tree

```text
<tasks-dir>/<task-id>/
  task.json
  events.jsonl
  videos/
    <video-id>/
      raw/
      input_labels/
      locany_labels/
      segments/
        seg_0000/
          raw/
          input_labels/
          locany_labels/
          tracking/
          package/
          reviewed/
          exports/
  tracking/    # temporary full-video compatibility path
  package/     # temporary full-video compatibility path
  reviewed/    # temporary full-video compatibility path
  exports/     # temporary full-video compatibility path
```

Artifacts are stage-specific and should not overwrite uploaded source labels or reviewed results.

## YOLO Input And Output

One TXT file represents one video frame. Accepted rows are:

```text
class_id x_center y_center width height
class_id x_center y_center width height score
```

Coordinates and dimensions are normalized. LocateAnything output includes score. Class IDs are non-negative integers and must match the task class table.

## Tracking Results

`tracking_results.json` is the exchange format between the MOT pipeline, Annotator and SAM3.1. It contains video metadata and tracks with stable `track_id`, `class_id` and per-frame pixel-space bounding boxes. Treat it as the authoritative editable trajectory representation; derive YOLO files from it after review.

## ZIP Portability

ZIP member paths use relative forward-slash names. Clients must not rely on Windows drive letters or Linux absolute paths inside packages. Extraction code rejects paths escaping the destination directory.

## Frame Numbering

Source labels commonly start at frame 1. Segment-local labels are renumbered from the configured frame offset. Always validate the first and last frame after splitting a new data source, especially when upstream models use zero-based filenames.
