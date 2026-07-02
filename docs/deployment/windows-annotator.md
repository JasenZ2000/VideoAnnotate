# Employee Annotator Setup

Annotator is designed to run on each employee's Windows PC so frame navigation and box editing do not depend on network latency.

## Install From Source

```powershell
cd D:\video-annotation-workflow
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements\annotator.txt
.\scripts\windows\run-annotator.bat
```

Open `http://127.0.0.1:7860`. Keep the bind host at `127.0.0.1` unless other PCs genuinely need to access this employee instance.

## Workspace Configuration

An annotation package/workspace contains video data, `tracking_results.json` and `config.json`. Start from `configs/annotator.example.json` and set:

- `sam31.server_url` to the GPU API;
- `video_transfer` to `path` for shared storage or `sftp` for upload;
- path prefixes for shared storage, or SFTP host/user/remote directory;
- `exports.class_labels` to the task class table.

For password authentication:

```powershell
$env:SAM31_SFTP_PASSWORD="<session-password>"
.\scripts\windows\run-annotator.bat
```

The password is read from the environment name in `sftp_password_env`; it should not be written into JSON or uploaded back to the platform. An SSH key is preferred where department policy allows it.

## SAM3.1 Usage

Open the workspace, select/create the intended track, save a box on the current frame and start `SAM31 Track Box`. The local server uploads or maps the video, submits an asynchronous job, polls it and merges returned boxes into the selected track.

Always inspect the generated continuation. SAM3.1 output is an editing aid, not an approval decision.
