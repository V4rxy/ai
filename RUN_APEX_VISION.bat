@echo off
cd /d "%~dp0"
py -3.11 -c "import PySide6, cv2, numpy, onnxruntime" >nul 2>nul
if errorlevel 1 (
  echo Installing requirements...
  py -3.11 -m pip uninstall -y onnxruntime
  py -3.11 -m pip install -r requirements.txt
  if errorlevel 1 ( echo Install failed. & pause & exit /b 1 )
)
py -3.11 main.py
pause
