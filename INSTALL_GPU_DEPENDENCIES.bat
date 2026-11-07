@echo off
setlocal
cd /d "%~dp0"
title Apex Vision - Install Dependencies

py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 was not found. Install Python 3.11 x64 first.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Installation complete.
echo Run CHECK_GPU.bat, then RUN_APEX_VISION.bat.
pause
exit /b 0

:failed
echo.
echo Installation failed. Copy the error text for troubleshooting.
pause
exit /b 1
