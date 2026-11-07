import sys
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception as e:
    print("preload_dlls note:", e)
print("Python:", sys.version)
print("ONNX Runtime:", ort.__version__)
prov = ort.get_available_providers()
print("Providers:", prov)
print("RESULT:", "CUDA available" if "CUDAExecutionProvider" in prov else "CUDA NOT available - app will use CPU")
