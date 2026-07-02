@echo off
setlocal
if "%ANNOTATOR_PORT%"=="" set ANNOTATOR_PORT=7860

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%ANNOTATOR_PORT% ^| findstr LISTENING') do (
  echo Stopping process %%a on port %ANNOTATOR_PORT%
  taskkill /F /PID %%a >nul 2>&1
)
