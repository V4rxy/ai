import sys

print("=== APEX VISION ENVIRONMENT CHECK ===")
print("Python:", sys.version.replace("\n", " "))

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except Exception as exc:
    print("OpenCV: NOT AVAILABLE -", exc)

try:
    import PySide6
    print("PySide6:", PySide6.__version__)
except Exception as exc:
    print("PySide6: NOT AVAILABLE -", exc)

try:
    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception as exc:
        print("ONNX Runtime DLL preload note:", exc)

    print("ONNX Runtime:", ort.__version__)
    providers = ort.get_available_providers()
    print("Execution providers:", providers)
    if "CUDAExecutionProvider" in providers:
        print("RESULT: CUDA provider is available.")
    else:
        print("RESULT: CUDA provider is unavailable; inference will use CPU.")
except Exception as exc:
    print("ONNX Runtime: NOT AVAILABLE -", exc)
    print("RESULT: Install dependencies with INSTALL_GPU_DEPENDENCIES.bat.")
