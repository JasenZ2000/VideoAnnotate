# SAM3.1 Server Tools

This directory contains the server-side tools used to run SAM3.1 from a ComfyUI
environment and return this project's `tracking_results.json` format.

The tools depend on the real server ComfyUI installation, not on copied Comfy
source files. Defaults are set in `sam31_track.py`:

- Comfy root: `/data2/DET_Group/ZZS/generate/update/ComfyUI`
- Checkpoint: `sam3.1_multiplex_fp16.safetensors`

Both can be overridden from the command line.

## Run the Remote Job Server

On the GPU server:

```bash
SAM31_PORT=9001 \
SAM31_CACHE_DIR=/data/cache/object-reid-sam31 \
SAM31_ALLOWED_ROOTS=/data/object-reid-clip \
./run_sam31_server.sh
```

The annotator sends only paths and bbox prompts to this server. Videos must be
available on the GPU server through shared storage or pre-sync. Use
`SAM31_ALLOWED_ROOTS` to restrict which video roots the API can read.

Health check:

```bash
curl http://localhost:9001/api/health
```

BBox job API:

```bash
curl -X POST http://localhost:9001/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/data/object-reid-clip/sampleInput/sample.mp4",
    "bbox": [524, 586, 816, 1070],
    "start_frame": 13,
    "max_frames": 1200,
    "class_id": 0,
    "device": "cuda:0"
  }'
```

Poll `/api/jobs/{job_id}` and fetch `/api/jobs/{job_id}/tracking-results` after
the job reaches `done`.

## Run SAM3.1

Text prompt detection + tracking:

```bash
python sam31_track.py \
  --video /path/to/video.mp4 \
  --prompt "person:5, car:3" \
  --out-dir /path/to/sam31_out \
  --max-frames 1200
```

BBox prompt tracking:

```bash
python sam31_track.py \
  --video /path/to/video.mp4 \
  --bbox 524,586,816,1070 \
  --start-frame 13 \
  --out-dir /path/to/sam31_out \
  --max-frames 1200
```

Multiple boxes can be passed by repeating `--bbox`.

Main output:

- `/path/to/sam31_out/tracking_results.json`

Debug output:

- `/path/to/sam31_out/sam31_detections.json`

## Append Results

Append a new SAM31 result into an existing annotator result while avoiding
`track_id` conflicts:

```bash
python append_tracking_results.py \
  --base /path/to/existing/tracking_results.json \
  --new /path/to/sam31_out/tracking_results.json \
  --output /path/to/merged_tracking_results.json
```

Use `--class-id` to override the class id of appended tracks.
