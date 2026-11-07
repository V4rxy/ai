# Apex Vision — NVIDIA GPU Edition (ViGEmBus & Closest-Target)

A high-performance local visual detection application for Apex Legends on Windows 11.

## Features
- NVIDIA RTX GPU Acceleration via ONNX Runtime CUDA
- Closest-Target Tracking: Automatically filters detections to track ONLY the enemy closest to screen center
- Virtual Dual-Joystick Visualizer with direction vector overlay
- ViGEmBus Virtual Controller Output (optional Xbox 360 controller output via vgamepad)
- Dark EngineX Gaming Interface with real-time FPS and GPU Latency monitoring

## Files Included
- `main.py`: Updated main Python application script
- `requirements.txt`: Python package dependencies (PySide6, OpenCV, NumPy, ONNX Runtime GPU, vgamepad)
- `INSTALL_GPU_DEPENDENCIES.bat`: Installs all required Python dependencies
- `CHECK_GPU.bat` & `check_gpu.py`: Verifies NVIDIA CUDA GPU acceleration
- `RUN_APEX_VISION.bat`: Application launcher script
- `README.txt`: Documentation and setup guide

## Setup Instructions
1. Run `INSTALL_GPU_DEPENDENCIES.bat` to install all requirements.
2. Run `CHECK_GPU.bat` to verify CUDA hardware acceleration.
3. Launch the app using `RUN_APEX_VISION.bat`.
4. Connect your PS5 capture card and select your model (`.onnx`).
5. (Optional) Check 'Enable ViGEmBus Virtual Controller Output' to pass right-stick aim vectors.
