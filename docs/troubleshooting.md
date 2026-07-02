# Troubleshooting

## A Long GPU Job Times Out Locally

Only submit/poll/download requests should have HTTP timeouts. Confirm the client received a `job_id` and is polling `/api/jobs/{id}`. Do not turn one inference into a single multi-hour HTTP response. Increase `download_timeout` only for result ZIP transfer.

## LocateAnything Runs Out Of GPU Memory In `generate()`

Model sharding or `max_memory` controls weight placement, not peak generation allocations. Split long videos, reduce `resize_long_edge`, reduce `max_new_tokens`, use the intended dtype and avoid concurrent jobs on the same model. Check for unrelated GPU processes before changing allocator settings.

## LocateAnything Fails On `str | Image.Image`

The environment is using Python 3.9 or older. Recreate it with Python 3.10+.

## Wrong CUDA Build

Install PyTorch from the official index matching the server driver/runtime before project dependencies. The root requirements intentionally do not install PyTorch. Verify with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## GPU Service Rejects A Video Path

The path must exist on Linux and be under `SAM31_ALLOWED_ROOTS` or `LOCANY_ALLOWED_ROOTS`. For shared storage, verify path-prefix mapping. For SFTP, verify upload directory, Linux account permissions and that the upload directory is allowed by the API.

## SFTP Asks For Login

SFTP always authenticates. Configure a username plus either an SSH key or the password environment variable named by `sftp_password_env`. The platform does not infer server credentials.

## Duplicate Tasks Appear

The current list endpoint deduplicates records by `task_id`. If duplicates remain, compare `task.json` IDs in the tasks directory and check whether an old process is serving a different `ANNOTATION_PLATFORM_TASKS_DIR`.

## Uploaded Filename Disappears In The Browser

Browser file inputs may clear after rerender. Confirm the selected file is retained in JavaScript state and inspect the network request. A successful upload appears in task detail and under the selected video's `raw/` directory.

## First Checks

Run `python scripts/check-services.py` with deployed URLs, inspect each service console, then inspect task `events.jsonl`. Remote job status responses include the final error message and bounded stdout/stderr where available.
