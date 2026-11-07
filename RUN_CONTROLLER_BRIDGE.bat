@echo off
title EngineX Chiaki Bridge
cd /d "%~dp0"
py -3.11 -m pip show PySide6 >nul 2>&1
if errorlevel 1 (
  echo PySide6 is missing. Installing...
  py -3.11 -m pip install PySide6
)
py -3.11 controller_bridge.py
if errorlevel 1 pause
