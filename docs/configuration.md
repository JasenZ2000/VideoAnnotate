# Configuration

Configuration is a JSON override merged onto `mot_pipeline.config.DEFAULT_CONFIG`. Unspecified fields retain code defaults.

## Files And Environment

- Root `config.json` is a safe local default with no real endpoints or credentials.
- `configs/platform.example.json` is the public-machine template.
- `configs/annotator.example.json` is the employee workspace template.
- `configs/gpu-services.env.example` lists GPU service environment variables.
- `*.local.json` is ignored by Git.

Platform config selection order is CLI `--config`, `ANNOTATION_PLATFORM_CONFIG`, then root `config.json`. Annotator uses workspace `config.json`; its project fallback can be overridden with `ANNOTATOR_CONFIG`.

## Remote Transfer Fields

| Field | Meaning |
| --- | --- |
| `server_url` | Base URL of the GPU service |
| `video_transfer` | `path` or `sftp` |
| `local_path_prefix` | Path visible to the Windows client |
| `remote_path_prefix` | Equivalent path visible to Linux |
| `sftp_host`, `sftp_port`, `sftp_username` | SFTP connection |
| `sftp_password_env` | Environment variable containing the password |
| `sftp_key_path` | Optional private key path |
| `sftp_remote_dir` | GPU-side upload directory, also allowed by the API |
| `sftp_reuse_existing` | Reuse an equal remote filename where supported |
| `request_timeout` | Timeout for short submit/poll requests, not total inference time |
| `poll_interval` | Seconds between job-status requests |

## GPU Service Environment

SAM3.1 uses `SAM31_*`; LocateAnything uses `LOCANY_*`. Important variables are service port, cache directory, model/checkpoint path, CUDA device, dtype and comma-separated allowed roots. See the environment example for names.

## Class Mapping

Platform task classes are stored as ordered ID/name pairs. LocateAnything receives class names as `categories` and a `class_map` such as:

```json
{
  "categories": ["person", "car"],
  "class_map": {"person": 0, "car": 1}
}
```

Use simple, distinct category names that the model is likely to return verbatim. Unknown labels currently fall back to the request's default class ID, so inspect multi-class samples before large production runs.
