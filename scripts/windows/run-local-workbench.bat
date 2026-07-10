@echo off
setlocal
cd /d "%~dp0\..\.."

if "%LOCAL_WORKBENCH_HOST%"=="" set LOCAL_WORKBENCH_HOST=127.0.0.1
if "%LOCAL_WORKBENCH_PORT%"=="" set LOCAL_WORKBENCH_PORT=7860

echo Starting local annotation workbench at http://%LOCAL_WORKBENCH_HOST%:%LOCAL_WORKBENCH_PORT%/
echo Annotator: http://%LOCAL_WORKBENCH_HOST%:%LOCAL_WORKBENCH_PORT%/annotator/
echo Frame sampler: http://%LOCAL_WORKBENCH_HOST%:%LOCAL_WORKBENCH_PORT%/sampler/
python -m local_workbench.server --host "%LOCAL_WORKBENCH_HOST%" --port "%LOCAL_WORKBENCH_PORT%" --open-browser %*
