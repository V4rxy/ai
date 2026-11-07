@echo off
cd /d "%~dp0"
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip uninstall -y onnxruntime
py -3.11 -m pip install --upgrade "onnxruntime-gpu[cuda,cudnn]" PySide6 opencv-python numpy
if errorlevel 1 (
  echo Installation failed. Copy the error text for troubleshooting.
  pause
  exit /b 1
)
echo Done. Run CHECK_GPU.bat to verify.
pause
