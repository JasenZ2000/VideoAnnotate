# Development

## Setup

```bash
python -m venv .venv
# Activate the environment for your shell.
pip install -r requirements/dev.txt
```

GPU model dependencies are deliberately separate. Most platform, annotator and MOT work should not require a GPU environment.

## Checks

```bash
python -m pytest
python -m compileall annotator workflow_platform mot_pipeline sam31
python scripts/check-services.py --platform http://127.0.0.1:8088
```

Use short synthetic videos and labels for integration testing; never commit recorded production data, task directories, credentials, model weights or generated archives.

## Change Boundaries

- Put task orchestration in `workflow_platform/`.
- Put interactive editing behavior in `annotator/`.
- Put reusable tracking and conversion logic in `mot_pipeline/`.
- Keep GPU HTTP contracts in `sam31/server.py` and `locateAnything/locateanything_video_server.py`.
- Prefer adapter changes over modifications to vendored LocateAnything model internals.

When changing a remote API, update its client, example config, API documentation and health/integration test in the same revision.

## Release Checklist

1. Run unit and compile checks.
2. Test platform create/upload/split on Windows.
3. Test one short LocateAnything segment and one SAM3.1 bbox job on Linux.
4. Verify class IDs in exported YOLO.
5. Confirm no `*.local.json`, credentials, videos or weights are staged.
6. Tag a known deployable commit and record environment versions used by each GPU service.
