@echo off
setlocal
cd /d "%~dp0\..\.."

if "%ANNOTATOR_HOST%"=="" set ANNOTATOR_HOST=127.0.0.1
if "%ANNOTATOR_PORT%"=="" set ANNOTATOR_PORT=7860

echo Starting local annotator at http://%ANNOTATOR_HOST%:%ANNOTATOR_PORT%
python -m annotator --host "%ANNOTATOR_HOST%" --port "%ANNOTATOR_PORT%" %*
