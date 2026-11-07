@echo off
setlocal
cd /d "%~dp0"
title Apex Vision - GPU Check

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" check_gpu.py
) else (
  py -3.11 check_gpu.py
)
echo.
pause
endlocal
