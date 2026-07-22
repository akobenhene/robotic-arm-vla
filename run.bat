@echo off
REM Run via venv Python (no Activate.ps1 required).
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Create with: py -3.11 -m venv .venv
  exit /b 1
)
if "%~1"=="" (
  ".venv\Scripts\python.exe" main.py --steps 400 --seed 0
) else (
  ".venv\Scripts\python.exe" %*
)
