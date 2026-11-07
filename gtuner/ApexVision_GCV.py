# Apex Vision: visual-only Gtuner IV Computer Vision worker.
# Place Apx8_Harlan.onnx in this folder. No controller input is read or sent.

import os
import json
import time
import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(ROOT, "apex_vision_config.json")


def read_config():
    cfg = {
        "model_path": "Apx8_Harlan.onnx",
        "confidence": 0.35,
        "iou": 0.45,
        "input_size": 640,
        "class_names": []
    }
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            cfg.update(loaded)
    except Exception:
        pass
    return cfg


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = None
        self.net = None
        self.backend = "not loaded"
        self.error = ""
        self.input_name = None
        self.iw = int(cfg.get("input_size", 640))
        self.ih = self.iw
        self.load()

    def load(self):
        path = self.cfg.get("model_path", "Apx8_Harlan.onnx")
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        if not os.path.isfile(path):
            self.error = "Model not found: " + path
            return

        if ort is not None:
            try:
                available = ort.get_available_providers()
                providers = []
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                if "CPUExecutionProvider" in available:
                    providers.append("CPUExecutionProvider")
                self.session = ort.InferenceSession(path, providers=providers or None)
                inp = self.session.get_inputs()[0]
                self.input_name = inp.name
                shape = inp.shape
                if len(shape) == 4:
                    if isinstance(shape[2], int):
                        self.ih = shape[2]
                    if isinstance(shape[3], int):
                        self.iw = shape[3]
                self.backend = "ONNX Runtime: " + ", ".join(self.session.get_providers())
                return
            except Exception as exc:
                self.error = "ONNX Runtime: " + str(exc)

        try:
            self.net = cv2.dnn.readNetFromONNX(path)
            self.backend = "OpenCV DNN CPU"
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.backend = "OpenCV DNN CUDA"
            except Exception:
                pass
            self.error = ""
        except Exception as exc:
            self.error = "Model load failed: " + str(exc)

    def infer(self, frame):
        if self.session is None and self.net is None:
            return [], self.error or "Model unavailable"

        fh, fw = frame.shape[:2]
        scale = min(self.iw / float(fw), self.ih / float(fh))
        rw, rh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (rw, rh))
        pad_x, pad_y = (self.iw - rw) // 2, (self.ih - rh) // 2
        canvas = np.full((self.ih, self.iw, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + rh, pad_x:pad_x + rw] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0,
                                     (self.iw, self.ih), swapRB=True)

        try:
            if self.session is not None:
                outs = self.session.run(None, {self.input_name: blob.astype(np.float32)})
            else:
                self.net.setInput(blob)
                outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            return self.parse(outs[0], fw, fh, scale, pad_x, pad_y), ""
        except Exception as exc:
            return [], "Inference error: " + str(exc)

    def parse(self, output, fw, fh, scale, pad_x, pad_y):
        arr = np.squeeze(np.asarray(output))
        if arr.ndim != 2:
            return []
        # Convert common [features, candidates] output into [candidates, features].
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 512:
            arr = arr.T

        boxes, scores, ids = [], [], []
        threshold = float(self.cfg.get("confidence", 0.35))
        for row in arr:
            if row.size < 6:
                continue
            # Supports common YOLOv8 (4 + classes) and YOLOv5 (5 + classes).
            class_scores = row[4:]
            cls = int(np.argmax(class_scores))
            score = float(class_scores[cls])
            # YOLOv5 objectness heuristic: multiply when a plausible objectness
            # value is present and doing so produces a stronger confidence estimate.
            if row.size >= 7 and 0.0 <= float(row[4]) <= 1.0:
                v5 = float(row[4]) * float(np.max(row[5:]))
                if v5 > score:
                    score = v5
                    cls = int(np.argmax(row[5:]))
            if not np.isfinite(score) or score < threshold:
                continue

            cx, cy, bw, bh = [float(v) for v in row[:4]]
            x1 = int(max(0, min(fw - 1, (cx - bw / 2 - pad_x) / scale)))
            y1 = int(max(0, min(fh - 1, (cy - bh / 2 - pad_y) / scale)))
            x2 = int(max(0, min(fw - 1, (cx + bw / 2 - pad_x) / scale)))
            y2 = int(max(0, min(fh - 1, (cy + bh / 2 - pad_y) / scale)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(score)
                ids.append(cls)

        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, threshold,
                                float(self.cfg.get("iou", 0.45)))
        names = self.cfg.get("class_names", [])
        result = []
        for k in np.asarray(keep).reshape(-1)[:50]:
            x, y, bw, bh = boxes[int(k)]
            cid = ids[int(k)]
            label = names[cid] if isinstance(names, list) and cid < len(names) else "class " + str(cid)
            result.append({
                "box": (x, y, x + bw, y + bh),
                "score": scores[int(k)],
                "label": str(label)
            })
        return result


class GCVWorker:
    def __init__(self, width, height):
        self.cfg = read_config()
        self.detector = Detector(self.cfg)
        self.frames = 0
        self.fps = 0.0
        self.fps_start = time.time()
        self.infer_ms = 0.0

    def process(self, frame, gcvdata):
        if frame is None:
            return frame, gcvdata

        out = frame.copy()
        h, w = out.shape[:2]
        center = (w // 2, h // 2)
        self.frames += 1
        now = time.time()
        if now - self.fps_start >= 1.0:
            self.fps = self.frames / (now - self.fps_start)
            self.frames = 0
            self.fps_start = now

        start = time.perf_counter()
        detections, error = self.detector.infer(frame)
        self.infer_ms = (time.perf_counter() - start) * 1000.0

        cv2.drawMarker(out, center, (255, 220, 0), cv2.MARKER_CROSS, 18, 2)
        nearest = None
        nearest_d2 = None

        for item in detections:
            x1, y1, x2, y2 = item["box"]
            target = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 90), 2)
            cv2.circle(out, target, 4, (0, 220, 255), -1)
            cv2.line(out, center, target, (255, 180, 0), 1)
            label = "{} {:.0f}%".format(item["label"], item["score"] * 100)
            cv2.putText(out, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 90), 1)
            d2 = (target[0] - center[0]) ** 2 + (target[1] - center[1]) ** 2
            if nearest_d2 is None or d2 < nearest_d2:
                nearest, nearest_d2 = (target, item), d2

        if nearest is not None:
            target, item = nearest
            cv2.line(out, center, target, (0, 255, 255), 3)
            dist = int(np.sqrt(nearest_d2))
            cv2.putText(out, "NEAREST 2D: {} px".format(dist),
                        (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)

        hud = "{} | {:.1f} FPS | {:.1f} ms | {} detections".format(
            self.detector.backend, self.fps, self.infer_ms, len(detections))
        cv2.rectangle(out, (5, 5), (min(w - 5, 900), 32), (20, 20, 20), -1)
        cv2.putText(out, hud[:110], (11, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (240, 240, 240), 1)
        if error:
            cv2.putText(out, error[:100], (12, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1)
        return out, gcvdata

    def __del__(self):
        self.detector = None
