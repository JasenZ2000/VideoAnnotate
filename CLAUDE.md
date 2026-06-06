# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-object tracking (MOT) pipeline that takes a video and per-frame YOLO detection labels, runs bidirectional tracking with Kalman-filtered motion models, fuses forward/backward tracklets, and outputs cropped clips per tracked identity plus a full-video overview with all bounding boxes drawn.

## Running the Pipeline

```bash
python main.py --video path/to/video.mp4 --label-dir path/to/labels --out-dir path/to/output --config config.json
```

Legacy entry point (`track_and_extract.py`) uses `--ann-dir` instead of `--label-dir`.

## Dependencies

- Python 3, OpenCV (`cv2`), NumPy (optional but enables Kalman filtering; falls back to simple velocity extrapolation without it).
- No package manager config (no requirements.txt/pyproject.toml) — install opencv-python and numpy manually.

## Architecture

The pipeline is orchestrated by `mot_pipeline/pipeline.py:run_pipeline()` which calls stages in sequence:

1. **Annotation loading** (`utils/io.py`) — reads YOLO txt files (one per frame, named `*_{frameindex}.txt`), converts normalized center-format boxes to pixel xyxy.
2. **Bidirectional tracking** (`tracking.py`) — runs the same tracker forward and backward over the frame sequence. Uses greedy IoU matching with a Kalman motion model (`MotionModel`). Produces `Tracklet` objects.
3. **Fusion** (`fusion.py`) — mutual-best-match between forward/backward tracklets using a UnionFind, then merges overlapping boxes per frame and applies sliding-window smoothing. Produces `FinalTrack` objects.
4. **Clip extraction** (`clips.py`) — computes fixed crop size per track, builds dense (interpolated) per-video-frame boxes, renders an overview video with all IDs, and extracts one cropped clip per track.
5. **Output writing** (`utils/io.py`) — writes `tracking_results.json` and `.csv`.
6. **Format conversion** (`utils/converters.py`) — exports tracking results to YOLO frame labels and Label Studio video-tracking JSON; also imports Label Studio JSON back to `tracking_results.json`.

## Key Design Patterns

- **Registries**: `TRACKER_REGISTRY` and `FUSION_REGISTRY` (dicts mapping method names to callables) allow swapping algorithms via the `"method"` config key without changing pipeline code.
- **Config layering**: `config.py:load_config()` deep-merges a user JSON over `DEFAULT_CONFIG`, so partial configs work — only override what you need.
- **Coordinate conventions**: internally all bounding boxes are pixel `[x1, y1, x2, y2]` (type alias `BBox = List[float]`). YOLO normalized center format is only used at I/O boundaries. Label Studio uses top-left percentage coordinates.
- **Clone before mutate**: `prepare_track_clips` destructively rewrites track frames to dense per-video-frame representation, so `pipeline.py` clones tracks before passing them in.

## Label Format Notes

- YOLO txt: `class cx cy w h` (normalized center).
- `tracking_results.json`: pixel xyxy, the canonical internal/output format.
- Label Studio video tracking: `x/y` as top-left percentages, `width/height` as percentages, 1-indexed frames, `enabled=false` marks sequence end.

## Adding a New Tracking or Fusion Method

1. Implement a function matching the signature of `run_tracking_pass` (for trackers) or `fuse_tracklets` (for fusion).
2. Register it in the corresponding `*_REGISTRY` dict.
3. Select it via the `"method"` key in the config JSON under `"tracking"` or `"fusion"`.
