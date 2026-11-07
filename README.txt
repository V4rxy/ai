# Apex Vision - EngineX Visual Detection

Windows 10/11 desktop application for viewing a PS5 capture-card feed,
running ONNX object detection, and displaying visual-only target direction.

## What is included

- main.py: EngineX desktop application.
- requirements.txt: Python dependencies.
- RUN_APEX_VISION.bat: creates a local virtual environment, installs
  dependencies, and launches the application.
- INSTALL_GPU_DEPENDENCIES.bat: installs dependencies into the local
  virtual environment without changing your system Python packages.
- CHECK_GPU.bat and check_gpu.py: reports ONNX Runtime providers.
- controller_bridge.py: separate Chiaki-ng companion window. It displays
  telemetry received on localhost UDP port 8765; the current main.py does
  not send telemetry to this companion yet.

## Recommended setup

1. Install 64-bit Python 3.11 and the Python Launcher for Windows.
2. Connect the PS5 HDMI output to your capture card, then connect the card
   to the Windows PC.
3. Double-click INSTALL_GPU_DEPENDENCIES.bat once.
4. Double-click CHECK_GPU.bat. Look for CUDAExecutionProvider in the
   provider list to confirm ONNX Runtime can see the NVIDIA GPU.
5. Double-click RUN_APEX_VISION.bat.
6. In the app, select the capture-card video device, click CONNECT CAPTURE,
   load a compatible ONNX model, then click START DETECTION.

## Important notes

- The app uses OpenCV VideoCapture with the Windows DirectShow backend for
  capture-card input. It does not use desktop duplication for the PS5 feed.
- Select the actual capture-card device from the device list. If it is absent,
  reconnect the USB cable, check the capture card driver, then click REFRESH.
- Load a model whose output layout is compatible with the parser in main.py.
  A .onnx extension alone does not guarantee model compatibility.
- CUDA availability depends on a compatible NVIDIA driver and the CUDA/cuDNN
  runtime dependencies installed by ONNX Runtime. If CUDA is unavailable,
  the app reports CPU fallback rather than claiming GPU acceleration.
- The right-stick display is visual guidance only. The app does not create,
  emulate, or inject controller inputs.
- The separate Chiaki companion can launch Chiaki-ng and display UDP telemetry,
  but telemetry sending from main.py still needs to be implemented separately.

## Troubleshooting

- If Python 3.11 is not found, install Python 3.11 x64 and enable the
  Python Launcher option.
- If installation fails, run INSTALL_GPU_DEPENDENCIES.bat again and keep
  the full terminal error.
- If the app opens but no capture devices appear, close other camera apps,
  reconnect the capture card, and press REFRESH.
- If the model fails to load, verify that it is a valid ONNX model and that
  its input/output tensor shapes match the model parser.
