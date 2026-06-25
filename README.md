# MOT Clip Extraction

Minimal, modular pipeline for:
- parsing YOLO frame annotations
- running bidirectional multi-object tracking
- fusing forward/backward tracklets
- smoothing trajectories
- extracting one cropped clip per final ID
- rendering one full-video tracking overview with all IDs drawn on the source video
- converting tracking outputs to YOLO frame labels and Label Studio video-tracking JSON

## Project Structure

- `main.py`: CLI entrypoint
- `mot_pipeline/config.py`: default parameters and JSON config loading
- `mot_pipeline/models.py`: shared data structures
- `mot_pipeline/tracking.py`: tracking logic and tracker registry
- `mot_pipeline/fusion.py`: fusion, smoothing, and final track building
- `mot_pipeline/clips.py`: clip sizing, dense boxes, and video extraction
- `mot_pipeline/utils/io.py`: annotation/video I/O and output writing
- `mot_pipeline/utils/bbox.py`: bbox conversion and geometry helpers
- `mot_pipeline/utils/converters.py`: tracking_results / YOLO / Label Studio conversions

## How To Run

Default parameters:

```bash
python main.py --video path/to/video.mp4 --label-dir path/to/labels --out-dir path/to/output
```

With a config file:

```bash
python main.py --video path/to/video.mp4 --label-dir path/to/labels --out-dir path/to/output --config config.json
```

Run the local annotator on a Linux server:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ANNOTATOR_PORT=7860 ./run_annotator.sh
```

You can also pass host/port directly:

```bash
python -m annotator --host 0.0.0.0 --port 7860
```

Annotator frame loading can be tuned per workspace in `config.json`:

```json
{
  "annotator": {
    "frame_buffer_ahead": 30,
    "frame_batch_size": 15,
    "frame_cache_limit": 80,
    "frame_batch_max": 30,
    "annotation_buffer_ahead": 60,
    "annotation_batch_size": 60,
    "annotation_cache_limit": 300,
    "annotation_batch_max": 200
  }
}
```

`frame_buffer_ahead` controls how many upcoming image frames the browser tries
to keep ready. `annotation_buffer_ahead` does the same for bbox overlays.
`frame_batch_size` and `annotation_batch_size` control normal fetch chunk sizes,
while `frame_batch_max` and `annotation_batch_max` are backend safety caps.

Example config:

```json
{
  "tracking": {
    "method": "sparse_track",
    "iou_match": 0.3,
    "max_missed": 15,
    "disable_kalman": false,
    "class_agnostic": false,
    "score_high": 0.5,
    "score_low": 0.1,
    "new_track_score": 0.6,
    "low_iou_match": 0.15
  },
  "fusion": {
    "iou_fuse": 0.5,
    "min_track_len": 10,
    "smooth_window": 5
  },
  "clips": {
    "pad_frames": 10,
    "crop_margin": 1.2,
    "crop_min_size": 128,
    "codec": "mp4v",
    "overview_filename": "tracking_overview.mp4",
    "overview_box_thickness": 2,
    "overview_font_scale": 0.7
  },
  "exports": {
    "export_yolo_from_tracking": true,
    "yolo_dirname": "tracking_yolo",
    "export_label_studio": true,
    "label_studio_filename": "tracking_results.label_studio.json",
    "label_studio_local_files_prefix": "/data/local-files/?d=",
    "class_labels": {
      "0": "Man"
    }
  }
}
```

## Outputs

- `tracking_results.json`: structured tracking result
- `tracking_results.csv`: flat tracking result
- `tracking_yolo/*.txt`: per-frame YOLO labels converted from `tracking_results.json`
- `tracking_results.label_studio.json`: Label Studio video-tracking JSON converted from `tracking_results.json`
- `clips/track_XXXX.mp4`: one cropped clip per final ID
- `tracking_overview.mp4`: full original video with all final IDs and boxes overlaid

## Where To Modify Things

- Change tracking behavior in `mot_pipeline/tracking.py`
- Change fusion behavior in `mot_pipeline/fusion.py`
- Change clip extraction behavior in `mot_pipeline/clips.py`
- Change default parameters in `mot_pipeline/config.py`
- Change format conversion behavior in `mot_pipeline/utils/converters.py`

The current refactor preserves the original algorithm and keeps the components replaceable through the small registries in `tracking.py` and `fusion.py`.

### Tracking Methods

Set `tracking.method` to one of:

- `iou_kalman`: original project tracker. It runs greedy IoU association over a constant-velocity / Kalman prediction.
- `sort`: alias for the original SORT-style IoU + Kalman pass.
- `bytetrack`: ByteTrack-style two-stage association. High-confidence detections are matched first, then unmatched tracks can recover from low-confidence detections.
- `oc_sort`: OC-SORT-style observation-centric association. It uses recent observed motion direction to recover from short occlusions and nonlinear motion.
- `bot_sort`: local BoT-SORT-inspired pass. It combines ByteTrack-style stages, confidence-fused IoU, and observation-gap velocity updates. Camera motion compensation and ReID require extra inputs and are not computed from plain YOLO txt labels.
- `hybrid_sort`: Hybrid-SORT-inspired pass. It adds weak cues from velocity direction, detection confidence, and bbox height consistency.
- `deep_oc_sort`: Deep OC-SORT-inspired pass. It uses the local motion/weak-cue fallback because this project does not currently parse or compute ReID embeddings.
- `cbiou`: C-BIoU-inspired cascaded buffered IoU tracker. It expands track/detection boxes with small and large buffers to recover irregular non-overlapping motion.
- `sparse_track`: SparseTrack-inspired pseudo-depth cascading tracker. It groups detections/tracks by bbox pseudo-depth and matches near-to-far to reduce crowded-scene ambiguity.

The parser accepts both `class cx cy w h` and `class cx cy w h confidence` YOLO lines. If confidence is missing, detections are treated as score `1.0`, so ByteTrack's low-confidence recovery only has an effect when detector scores are exported.

Example ByteTrack config:

```json
{
  "tracking": {
    "method": "bytetrack",
    "iou_match": 0.3,
    "max_missed": 30,
    "score_high": 0.5,
    "score_low": 0.1,
    "new_track_score": 0.6,
    "low_iou_match": 0.15,
    "disable_kalman": false
  }
}
```

### Fusion Methods

Set `fusion.method` to one of:

- `bidirectional_iou`: original conservative mutual-best forward/backward fusion.
- `forward_only`: use only forward tracklets as the final components.
- `backward_only`: use only backward tracklets, mainly for debugging.
- `bidirectional_iou_forward_primary`: keep forward tracklets as primary and merge only mutual-best backward matches.
- `bidirectional_iou_forward_unique`: keep forward tracklets, merge mutual-best backward matches, and keep only backward tracklets that are not duplicates of any forward tracklet.
- `bidirectional_iou_all_pairs`: merge every compatible forward/backward pair, more aggressive for fragmented tracks.
- `bidirectional_iou_nms`: run the original fusion and then suppress duplicate overlapping components.

See `FUSION_METHODS.md` for the trade-off table and suggested trial order.

## Remote SAM3.1-Assisted Annotation

The local annotator can start a SAM3.1 bbox-prompt tracking job from the active
frame:

1. Open a workspace in the annotator.
2. Select or create the target track.
3. Draw/save a bbox on the current frame.
4. Click `SAM31 Track Box` in the Annotation panel.

The annotator sends a lightweight job to a remote SAM31 FastAPI server. It does
not upload the video. The GPU server reads the video through shared storage,
runs `sam31/sam31_track.py`, and returns `tracking_results.json`. When the job
finishes, the generated boxes are written back to the same active track from the
prompt frame onward, and the workspace `tracking_results.json` is refreshed.

Start the remote SAM31 service on the GPU server:

```bash
SAM31_PORT=9001 \
SAM31_CACHE_DIR=/data/cache/object-reid-sam31 \
SAM31_ALLOWED_ROOTS=/data/object-reid-clip \
./run_sam31_server.sh
```

Configure each annotator workspace `config.json`:

```json
{
  "sam31": {
    "runner": "remote",
    "server_url": "http://gpu-server:9001",
    "local_path_prefix": "Z:/object-reid-clip",
    "remote_path_prefix": "/data/object-reid-clip",
    "poll_interval": 2
  }
}
```

`local_path_prefix` and `remote_path_prefix` describe the same shared storage
from the annotator machine and the GPU server. If the annotator is already using
server paths, leave both prefix fields empty. The optional `GPU` field in the
annotator overrides the configured device by passing `cuda:<n>` to the remote
job; leave it empty to use the server default.

If the GPU server cannot see the same path, use SFTP transfer instead:

```json
{
  "sam31": {
    "runner": "remote",
    "server_url": "http://gpu-server:9001",
    "video_transfer": "sftp",
    "sftp_host": "gpu-server",
    "sftp_port": 22,
    "sftp_username": "your-user",
    "sftp_password_env": "SAM31_SFTP_PASSWORD",
    "sftp_key_path": "",
    "sftp_remote_dir": "/data2/DET_Group/ZZS/my_sam3/tmp/videos",
    "sftp_reuse_existing": true
  }
}
```

Set the password outside config:

```bash
export SAM31_SFTP_PASSWORD='your-password'
```

or use `sftp_key_path` for key-based SSH auth. The SAM31 server must allow the
upload directory, for example:

```bash
SAM31_ALLOWED_ROOTS=/data2/DET_Group/ZZS/my_sam3/tmp ./run_sam31_server.sh
```

The annotator loads the project root `config.json` first, then overlays the
opened workspace or segment `config.json`. Existing workspaces that do not have
a `sam31` section will therefore inherit the root SAM31 server settings.

## Remote LocateAnything YOLO Proposal

The annotator can also submit the opened workspace video to a remote
LocateAnything service. The remote service runs prompt-based grounding on video
frames, writes YOLO txt labels, zips the result, and the local annotator
downloads it into a separate run folder. It does not merge with existing labels
yet.

Start the remote LocateAnything service on the GPU server:

```bash
LOCANY_PORT=9011 \
LOCANY_CACHE_DIR=/data/cache/object-reid-locany \
LOCANY_ALLOWED_ROOTS=/data2/DET_Group/ZZS/locateanything/tmp \
./run_locateanything_server.sh
```

Configure `config.json`:

```json
{
  "locateanything": {
    "server_url": "http://gpu-server:9011",
    "video_transfer": "sftp",
    "sftp_host": "gpu-server",
    "sftp_username": "your-user",
    "sftp_password_env": "SAM31_SFTP_PASSWORD",
    "sftp_remote_dir": "/data2/DET_Group/ZZS/locateanything/tmp/videos",
    "device": "cuda",
    "dtype": "bf16",
    "resize_long_edge": 1024,
    "generation_mode": "slow",
    "max_new_tokens": 512,
    "frame_offset": 1,
    "class_id": 0,
    "score": 1.0
  }
}
```

In the annotator sidebar, enter a prompt in `LocateAnything YOLO` and click
`Remote Infer YOLO`. Results are saved under:

```text
<workspace>/locateanything_runs/<job_id>/<video_stem>/<video_stem>_1.txt
```

There is one txt per video frame. Empty frames are represented by empty txt
files. Each non-empty line uses:

```text
class x_center y_center width height score
```

SAM31 bbox post-processing can interpolate short area/size spikes caused by mask
outliers:

```json
{
  "sam31": {
    "postprocess_spikes": true,
    "spike_area_ratio": 4.0,
    "spike_size_ratio": 3.0,
    "spike_history": 10,
    "spike_min_history": 3,
    "spike_max_run": 10
  }
}
```

This runs automatically when SAM31 results are merged into the active track.
The annotator also has a `Fix Spikes` button in the Interpolation panel for
repairing an existing track manually.

## Label Format Notes

- YOLO txt uses `class cx cy w h` with normalized center coordinates. An optional sixth `confidence` column is supported for ByteTrack-style methods.
- `tracking_results.json` is the dense internal/result format used by this project.
- Label Studio video tracking uses:
  - top-level `video` path like `"/data/local-files/?d=" + relative_video_path`
  - `box[].sequence[].x/y` as top-left percentages, not center coordinates
  - `width/height` as percentage values
  - `enabled=false` as the terminal keyframe for that sequence

The converter module supports:
- `tracking_results.json -> YOLO`
- `tracking_results.json -> Label Studio JSON`
- `Label Studio JSON -> tracking_results.json`
