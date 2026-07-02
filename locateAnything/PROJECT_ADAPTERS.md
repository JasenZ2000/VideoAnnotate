# Project LocateAnything Adapters

Most of this directory is vendored LocateAnything research code. The annotation workflow adds a small adapter layer:

- `locateanything_worker.py`: model loading and task wrappers.
- `debug_infer.py`: one-image inference diagnostics and visualization.
- `batch_yolo_infer.py`: resumable image-tree to YOLO inference with sharding and class mapping.
- `serve_locateanything.py`: legacy one-image HTTP debug service.
- `check_image_server.py`: client for the legacy one-image service.
- `locateanything_video_server.py`: asynchronous video-to-YOLO service used by the workflow platform.

Start the production-facing video service from the repository root:

```bash
./scripts/linux/run-locateanything-server.sh
```

The video API exposes `/api/health`, `/api/jobs`, job polling and YOLO ZIP download. It serializes model execution in one process; scale by running explicit GPU-bound service instances, not by sending concurrent generation calls to one worker.

When updating upstream LocateAnything, preserve these adapter files separately and re-run direct image inference before testing the video service.
