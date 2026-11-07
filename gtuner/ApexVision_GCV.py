"""
Apex Vision Gtuner IV / Titan Two Computer Vision worker.

Visual-only: detects objects from the Gtuner CV frame and draws a right-stick
direction guide. It deliberately leaves gcvdata unchanged and never writes
controller inputs.
"""

from pathlib import Path
import json
import math

import cv2
import numpy as np


class GCVWorker:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.net = None
        self.model_error = ""
        self.class_names = []
        self.input_size = 640
        self.confidence = 0.40
        self.nms_threshold = 0.45
        self.last_target = None
        self.frame_number = 0

        self.base_dir = Path(__file__).resolve().parent
        self.config_path = self.base_dir / "apex_vision_config.json"
        self._load_config()

    def _load_config(self):
        defaults = {
            "model_path": "",
            "names_path": "",
            "input_size": 640,
            "confidence": 0.40,
            "nms_threshold": 0.45,
            "use_cuda": False
        }
        try:
            if self.config_path.is_file():
                with self.config_path.open("r", encoding="utf-8") as handle:
                    user_config = json.load(handle)
                if isinstance(user_config, dict):
                    defaults.update(user_config)
        except Exception as exc:
            self.model_error = "Config error: " + str(exc)

        self.input_size = max(160, min(1280, int(defaults["input_size"])))
        self.confidence = max(0.05, min(0.95, float(defaults["confidence"])))
        self.nms_threshold = max(0.05, min(0.95, float(defaults["nms_threshold"])))

        names_path = self._resolve_path(defaults["names_path"])
        if names_path and names_path.is_file():
            try:
                self.class_names = [
                    line.strip() for line in names_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines() if line.strip()
                ]
            except Exception as exc:
                self.model_error = "Names file error: " + str(exc)

        model_path = self._resolve_path(defaults["model_path"])
        if not model_path or not model_path.is_file():
            self.model_error = (
                "Set model_path in gtuner/apex_vision_config.json"
            )
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # CUDA is optional. Standard pip OpenCV wheels are usually CPU-only.
            if bool(defaults.get("use_cuda", False)):
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                except Exception:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self.model_error = ""
        except Exception as exc:
            self.net = None
            self.model_error = "ONNX load failed: " + str(exc)

    def _resolve_path(self, value):
        if not value:
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def _detect(self, frame):
        if self.net is None:
            return []

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        output = self.net.forward()
        rows = np.asarray(output)

        while rows.ndim > 2:
            rows = rows[0]
        if rows.ndim != 2 or rows.size == 0:
            return []

        # Common YOLOv8 export: [features, candidates]. Convert to rows.
        if rows.shape[0] < rows.shape[1] and rows.shape[0] <= 256:
            rows = rows.T

        boxes = []
        scores = []
        class_ids = []

        for row in rows:
            if row.shape[0] < 5:
                continue

            # YOLOv5-style exports include objectness at index 4.
            if row.shape[0] >= 85:
                class_scores = row[5:]
                class_id = int(np.argmax(class_scores))
                score = float(row[4]) * float(class_scores[class_id])
            else:
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])

            if not np.isfinite(score) or score < self.confidence:
                continue

            cx, cy, box_w, box_h = map(float, row[:4])
            if max(abs(cx), abs(cy), abs(box_w), abs(box_h)) <= 2.0:
                cx *= self.input_size
                cy *= self.input_size
                box_w *= self.input_size
                box_h *= self.input_size

            x1 = int((cx - box_w / 2.0) * width / self.input_size)
            y1 = int((cy - box_h / 2.0) * height / self.input_size)
            box_w = int(box_w * width / self.input_size)
            box_h = int(box_h * height / self.input_size)

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            box_w = max(0, min(width - x1, box_w))
            box_h = max(0, min(height - y1, box_h))
            if box_w < 2 or box_h < 2:
                continue

            boxes.append([x1, y1, box_w, box_h])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        kept = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence, self.nms_threshold
        )
        if len(kept) == 0:
            return []

        detections = []
        for index in np.asarray(kept).reshape(-1):
            i = int(index)
            detections.append((boxes[i], scores[i], class_ids[i]))
        return detections

    def _draw_stick_guide(self, frame, dx, dy, distance):
        height, width = frame.shape[:2]
        radius = max(34, min(58, height // 10))
        cx = width - radius - 24
        cy = height - radius - 26

        cv2.circle(frame, (cx, cy), radius, (65, 75, 95), 2, cv2.LINE_AA)
        cv2.line(frame, (cx - radius, cy), (cx + radius, cy),
                 (65, 75, 95), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - radius), (cx, cy + radius),
                 (65, 75, 95), 1, cv2.LINE_AA)

        nx = max(-1.0, min(1.0, dx / max(1.0, width / 2.0)))
        ny = max(-1.0, min(1.0, dy / max(1.0, height / 2.0)))
        px = int(cx + nx * radius * 0.78)
        py = int(cy + ny * radius * 0.78)

        cv2.arrowedLine(frame, (cx, cy), (px, py), (118, 185, 0),
                        3, cv2.LINE_AA, tipLength=0.28)
        cv2.circle(frame, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT STICK GUIDE - MANUAL",
                    (max(8, cx - 150), max(18, cy - radius - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (118, 185, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, "YOU MOVE THE STICK",
                    (max(8, cx - 105), min(height - 6, cy + radius + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 225, 235), 1,
                    cv2.LINE_AA)

    def process(self, frame, gcvdata):
        if frame is None or not hasattr(frame, "shape"):
            return frame, gcvdata

        output = frame.copy()
        height, width = output.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.drawMarker(output, (center_x, center_y), (150, 150, 150),
                       cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)

        self.frame_number += 1

        if self.net is None:
            message = self.model_error or "ONNX model not loaded"
            cv2.putText(output, "APEX VISION - MODEL NOT READY",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(output, message[:100], (16, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (230, 230, 230), 1, cv2.LINE_AA)
            return output, gcvdata

        try:
            detections = self._detect(output)
            best = None
            best_distance = float("inf")

            for box, score, class_id in detections:
                x, y, box_w, box_h = box
                target_x = x + box_w // 2
                target_y = y + box_h // 2
                distance = math.hypot(target_x - center_x, target_y - center_y)
                if distance < best_distance:
                    best_distance = distance
                    best = (box, score, class_id, target_x, target_y)

            cv2.putText(output, "APEX VISION - GTUNER CV",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (118, 185, 0), 2, cv2.LINE_AA)

            if best is None:
                cv2.putText(output, "NO DETECTION",
                            (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (190, 190, 190), 1, cv2.LINE_AA)
            else:
                box, score, class_id, target_x, target_y = best
                x, y, box_w, box_h = box
                label = (
                    self.class_names[class_id]
                    if 0 <= class_id < len(self.class_names)
                    else "Class " + str(class_id)
                )

                cv2.rectangle(output, (x, y), (x + box_w, y + box_h),
                              (118, 185, 0), 2)
                cv2.putText(output, "{} {:.0f}%".format(label, score * 100),
                            (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (118, 185, 0), 2, cv2.LINE_AA)
                cv2.line(output, (center_x, center_y),
                         (target_x, target_y), (0, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(output, (target_x, target_y), 5,
                           (0, 220, 255), -1, cv2.LINE_AA)

                dx = float(target_x - center_x)
                dy = float(target_y - center_y)
                self._draw_stick_guide(output, dx, dy, best_distance)

                cv2.putText(
                    output,
                    "OFFSET X:{:+.0f} Y:{:+.0f}  DIST:{:.0f}px".format(
                        dx, dy, best_distance
                    ),
                    (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 220, 255), 1, cv2.LINE_AA
                )

        except Exception as exc:
            cv2.putText(output, "DETECTION ERROR: " + str(exc)[:80],
                        (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 255), 1, cv2.LINE_AA)

        # Preserve Gtuner's data buffer exactly. No controller commands are sent.
        return output, gcvdata
