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

## Titan Two / Gtuner IV Computer Vision version

A separate visual-only worker is included for running inside Gtuner IV's
Computer Vision panel:

- gtuner/ApexVision_GCV.py: GCVWorker that receives Gtuner's video frame,
  runs an ONNX model through OpenCV DNN, highlights the closest detection,
  and draws a manual right-stick direction guide.
- gtuner/apex_vision_config.json: model configuration template.

Setup:

1. Connect the Titan Two PROG port to the PC and open Gtuner IV.
2. In Gtuner IV Preferences -> Computer Vision, select the Python environment
   that can import both cv2 and numpy.
3. Open the Computer Vision panel and select the PS5 capture-card video input.
4. Open gtuner/ApexVision_GCV.py as a GCV script and start it in the CV panel.
5. Edit gtuner/apex_vision_config.json and set model_path to the full path of
   your ONNX model. Optionally set names_path to a text file with one class
   name per line. Restart the GCV script after changing the config.
6. Start with use_cuda=false. Set it true only if your OpenCV build includes
   CUDA support; ordinary pip OpenCV wheels generally use the CPU.

This GCV worker uses Gtuner's frame callback. It does not open a second video
capture device, does not require the standalone app's UDP bridge, and returns
Gtuner's gcvdata buffer unchanged. It does not write stick values, emulate a
controller, or inject inputs. The Titan Two remains responsible for the
physical controller passthrough; the direction indicator is for the player
to follow manually.

Model compatibility: the worker expects a YOLO-style ONNX output with
center-x, center-y, width, height, and class scores (common YOLOv8 exports;
YOLOv5 exports with objectness are also handled). Other ONNX output layouts
may need a model-specific parser. Gtuner's embedded Python/OpenCV versions
and actual Titan Two/capture-card hardware have not been tested in this
repository, so verify the CV panel loads the worker before relying on it.

## Troubleshooting

- If Python 3.11 is not found, install Python 3.11 x64 and enable the
  Python Launcher option.
- If installation fails, run INSTALL_GPU_DEPENDENCIES.bat again and keep
  the full terminal error.
- If the app opens but no capture devices appear, close other camera apps,
  reconnect the capture card, and press REFRESH.
- If the model fails to load, verify that it is a valid ONNX model and that
  its input/output tensor shapes match the model parser.
