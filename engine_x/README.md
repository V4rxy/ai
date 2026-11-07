# Engine X

A native Windows desktop workspace for Titan Two device diagnostics and Computer Vision.

## First milestone

- Qt 6 dark desktop dashboard
- OpenCV capture-device discovery and live preview
- ONNX model loading and metadata inspection (when ONNX Runtime is installed)
- Clear diagnostics for missing dependencies
- No automated controller inputs or aiming features

## Build requirements

- Windows 10/11
- Visual Studio 2022 with the Desktop development with C++ workload
- CMake 3.24+
- Qt 6.5+
- OpenCV 4.x with videoio support
- ONNX Runtime C++ package (optional for the first UI/capture milestone)

## Configure and build

Run from a Visual Studio Developer PowerShell:

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64"
cmake --build build --config Release
```

Make sure CMake can locate OpenCV and, if installed, ONNX Runtime. Package paths vary by installation.

## Capture cards

The Capture tab lists OpenCV camera/video input indices. Select the index corresponding to the capture card. Some capture cards expose more than one format; format selection and latency tuning will be added after testing with the actual device.

## Titan Two integration

This first milestone does not pretend to implement an undocumented USB protocol. The Device panel is an integration placeholder. The next step is to use documented Gtuner/Titan Two interfaces or SDK functionality and verify device communication against the user's installed Gtuner IV version.

## ONNX model support

The current model inspector loads an ONNX file and reports its input/output metadata when ONNX Runtime is available. It does not assume a YOLO output layout. A later inference stage must be matched to the actual model's tensor names, dimensions, and preprocessing requirements.

## Status

Starter project; not yet built or tested on the target Windows machine.
