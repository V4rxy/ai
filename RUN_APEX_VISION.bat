@echo off
setlocal
title Apex Vision - EngineX
cd /d "%~dp0"

echo ==========================================
echo   APEX VISION - WINDOWS LAUNCHER
echo ==========================================

py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 was not found.
  echo Install Python 3.11 x64 and enable the Python Launcher.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating project virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)

echo Checking and installing project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo Could not update pip.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check the messages above.
  pause
  exit /b 1
)

echo Starting Apex Vision...
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
  echo Apex Vision exited with an error.
  pause
)

endlocal
