@echo off
:: One-click launcher for the video annotator (Windows)
:: Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
::
:: Workspace example: D:\projects\object-reid-clip\sampleInput
::   Contains: sample.mp4 + sample\ (YOLO labels folder)

where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

echo Starting Video Annotator on http://localhost:7860
uv run annotator
