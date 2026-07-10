@echo off
setlocal
if "%LOCAL_WORKBENCH_PORT%"=="" set LOCAL_WORKBENCH_PORT=7860

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%LOCAL_WORKBENCH_PORT% ^| findstr LISTENING') do (
  echo Stopping process %%a on port %LOCAL_WORKBENCH_PORT%
  taskkill /F /PID %%a >nul 2>&1
)
