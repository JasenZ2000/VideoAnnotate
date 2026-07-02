# Windows Platform Deployment

## Prerequisites

- Windows 10/11 or Windows Server
- Python 3.10 or newer (3.11 recommended)
- FFmpeg/OpenCV-compatible video codecs
- A data volume with enough space for source videos, segment copies and exported packages
- Network access to the Linux GPU service ports and, for SFTP mode, port 22

## Install

```powershell
git clone <repository-url> D:\video-annotation-workflow
cd D:\video-annotation-workflow
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\platform.txt
Copy-Item configs\platform.example.json configs\platform.local.json
```

Edit `configs\platform.local.json` with the LocateAnything URL and transfer settings. Do not commit this file.

For password-based SFTP, set the password only in the service account environment:

```powershell
[Environment]::SetEnvironmentVariable(
  "LOCANY_SFTP_PASSWORD",
  "<password>",
  "User"
)
```

Prefer an SSH key and a dedicated restricted Linux account for unattended operation.

## Run

```powershell
$env:ANNOTATION_PLATFORM_TASKS_DIR="D:\annotation_tasks"
$env:ANNOTATION_PLATFORM_CONFIG="$PWD\configs\platform.local.json"
.\scripts\windows\run-platform.bat
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8088/api/health
```

Allow inbound TCP 8088 only from the department network. For persistent operation, run the command under a Windows service wrapper or Task Scheduler using a dedicated service account with read/write access to the task volume.

## Backup

Back up the entire tasks directory. Each task is filesystem-contained (`task.json`, events and artifacts), so no separate database dump is required in the MVP. Avoid backing up while a large upload or ZIP extraction is actively writing; use volume snapshots when available.
