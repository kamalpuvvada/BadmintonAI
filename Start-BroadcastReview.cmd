@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The BadmintonAI Python environment is missing.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" review_broadcast.py %*
