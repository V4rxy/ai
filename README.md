# Apex Vision — Gtuner IV CV starter

This branch adds a visual-only Gtuner IV Computer Vision worker.

## Files
- gtuner/ApexVision_GCV.py — receives video frames from Gtuner IV and draws ONNX detections.
- gtuner/apex_vision_config.json — model path and detection settings.
- gtuner/requirements.txt — Python dependencies.

## Setup
1. Configure Python and OpenCV in Gtuner IV Computer Vision Preferences.
2. Install the requirements using the same Python interpreter selected in Gtuner IV.
3. Put your ONNX model in the gtuner folder as Apx8_Harlan.onnx, or edit model_path in the JSON config.
4. In the Gtuner IV Computer Vision panel, add/select your capture-card video input.
5. Open ApexVision_GCV.py as the GCV script and start it.

## Expected preview
- Boxes and class/confidence labels for detected objects.
- Lines from the frame center to each detected object's center.
- A highlighted detection with the smallest 2D screen-space distance.
- FPS, inference time, and the selected inference backend.

The highlighted line is a straight 2D screen-space guide, not a 3D route. This worker does not read or send controller inputs.

## GPU note
The worker prefers ONNX Runtime CUDA when a compatible CUDA Execution Provider is installed. It falls back to ONNX Runtime CPU or OpenCV DNN. A standard pip OpenCV wheel is generally CPU-only. The displayed backend indicates what actually loaded.

## Model compatibility
The parser supports common YOLOv5 and YOLOv8 output layouts, but ONNX models differ. If this model uses another tensor layout or preprocessing, its exact input/output shapes must be inspected and the parser adjusted. The ONNX model is not included.

This is a starter integration and has not been verified on your specific Gtuner IV installation or capture card.
