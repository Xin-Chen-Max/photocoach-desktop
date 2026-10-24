@echo off
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo PhotoCoach's local Python environment is missing. See README.md for setup.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m photocoach %*
if errorlevel 1 pause
