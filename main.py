import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout,
    QWidget, QCheckBox
)

# Optional ViGEmBus controller output integration (vgamepad)
try:
    import vgamepad as vg
    VIGEM_AVAILABLE = True
except ImportError:
    vg = None
    VIGEM_AVAILABLE = False

APP_NAME = "APEX VISION - NVIDIA GPU EDITION (ViGEmBus)"
INPUT_SIZE = 640

# Make pip-installed CUDA/cuDNN DLLs discoverable
try:
    ort.preload_dlls()
except Exception:
    pass


class JoystickVisualizer(QWidget):
    """
    Custom Dual Joystick Visualizer displaying virtual sticks, direction vectors,
    and relative target position markers derived from screen-center calculations.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMaximumHeight(220)
        
        # Target tracking state
        self.dx = 0.0
        self.dy = 0.0
        self.dist = 0.0
        self.angle_deg = 0.0
        self.has_target = False

    def update_target(self, dx, dy, dist, angle_deg, has_target):
        self.dx = dx
        self.dy = dy
        self.dist = dist
        self.angle_deg = angle_deg
        self.has_target = has_target
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        painter.fillRect(rect, QColor("#111522"))
        painter.setPen(QPen(QColor("#22293d"), 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)

        w = rect.width()
        h = rect.height()
        cy = h / 2.0
        
        radius = min(h * 0.35, 60.0)
        left_cx = w * 0.25
        right_cx = w * 0.75

        # Render Joysticks
        self._draw_joystick_base(painter, left_cx, cy, radius, "LEFT STICK (AIM GUIDANCE)")
        self._draw_joystick_base(painter, right_cx, cy, radius, "RIGHT STICK (TARGET MAGNITUDE)")
        
        self._draw_center_info(painter, w / 2.0, cy)

        if self.has_target:
            max_offset = 640.0 / 2.0
            norm_dx = max(-1.0, min(1.0, self.dx / max_offset))
            norm_dy = max(-1.0, min(1.0, self.dy / max_offset))
            
            target_x = right_cx + norm_dx * (radius * 0.85)
            target_y = cy + norm_dy * (radius * 0.85)

            # Directional ray on Right Joystick
            painter.setPen(QPen(QColor("#76b900"), 2, Qt.DashLine))
            painter.drawLine(QPointF(right_cx, cy), QPointF(target_x, target_y))

            # Target Marker on Right Joystick
            painter.setBrush(QBrush(QColor("#76b900")))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(QPointF(target_x, target_y), 8, 8)
            
            # Left Joystick Direction Vector Line
            rad = np.radians(self.angle_deg)
            dir_x = left_cx + np.cos(rad) * (radius * 0.75)
            dir_y = cy + np.sin(rad) * (radius * 0.75)
            painter.setPen(QPen(QColor("#00e676"), 2))
            painter.drawLine(QPointF(left_cx, cy), QPointF(dir_x, dir_y))
            painter.setBrush(QBrush(QColor("#00e676")))
            painter.drawEllipse(QPointF(dir_x, dir_y), 6, 6)
        else:
            self._draw_stick_head(painter, left_cx, cy, 14, QColor("#3b4259"))
            self._draw_stick_head(painter, right_cx, cy, 14, QColor("#3b4259"))

    def _draw_joystick_base(self, painter, cx, cy, radius, label):
        painter.setPen(QPen(QColor("#76b900"), 1.5))
        painter.setBrush(QBrush(QColor("#181f30")))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        
        painter.setPen(QPen(QColor("#2d364f"), 1, Qt.DotLine))
        painter.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))
        painter.drawLine(QPointF(cx, cy - radius), QPointF(cx, cy + radius))
        
        painter.setBrush(QBrush(QColor("#45506b")))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 3, 3)

        painter.setPen(QPen(QColor("#8d93a8")))
        font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(cx - 100, cy + radius + 6, 200, 20), Qt.AlignCenter, label)

    def _draw_stick_head(self, painter, cx, cy, radius, color):
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor("#5c6682"), 1.5))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

    def _draw_center_info(self, painter, cx, cy):
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        if self.has_target:
            painter.setPen(QPen(QColor("#76b900")))
            painter.drawText(QRectF(cx - 120, cy - 40, 240, 20), Qt.AlignCenter, "CLOSEST TARGET ACQUIRED")
            
            painter.setFont(QFont("Consolas", 10))
            painter.setPen(QPen(QColor("#e9eaf2")))
            offset_text = f"X: {self.dx:+.1f} px  |  Y: {self.dy:+.1f} px"
            dist_text = f"Distance: {self.dist:.1f} px  |  Dir: {self.angle_deg:.1f}°"
            painter.drawText(QRectF(cx - 150, cy - 15, 300, 20), Qt.AlignCenter, offset_text)
            painter.drawText(QRectF(cx - 150, cy + 10, 300, 20), Qt.AlignCenter, dist_text)
        else:
            painter.setPen(QPen(QColor("#646e85")))
            painter.drawText(QRectF(cx - 120, cy - 15, 240, 20), Qt.AlignCenter, "NO TARGET DETECTED")
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(QRectF(cx - 120, cy + 5, 240, 20), Qt.AlignCenter, "Searching screen center...")


class VisionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apex Vision — NVIDIA RTX 5060 Ti (ViGEmBus Edition)")
        self.resize(1340, 880)
        self.setMinimumSize(1024, 720)

        self.capture = None
        self.session = None
        self.input_name = None
        self.input_size = (INPUT_SIZE, INPUT_SIZE)
        self.active_device = "CPU (Idle)"
        self.model_path = None
        self.class_names = []
        self.running = False
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0
        self.frame_count = 0
        self.last_inference_ms = 0.0
        self.conf_value = 0.45
        self.nms_value = 0.45
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_detections = []
        self.alive = False
        self.cap_thread = None
        self.infer_thread = None
        self.shown_frame_id = -1

        # ViGEmBus Virtual Controller Initialization
        self.gamepad = None
        self.vigem_enabled = False
        if VIGEM_AVAILABLE:
            try:
                self.gamepad = vg.VX360Gamepad()
                print("[ViGEmBus]: Xbox 360 Virtual Controller initialized successfully.")
            except Exception as e:
                print(f"[ViGEmBus Init Warning]: {e}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self._build_ui()
        self._apply_style()
        self._check_gpu_availability()
        self.refresh_devices()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)

        # Header
        header = QHBoxLayout()
        brand_box = QVBoxLayout()
        title = QLabel("<span style='color:#ffffff; font-weight:900;'>ENGINE</span><span style='color:#76b900; font-weight:900;'>X</span> <span style='color:#8d93a8; font-weight:400; font-size:18px;'>| APEX VISION (ViGEmBus & CLOSEST TARGET FILTER)</span>")
        title.setObjectName("brand")
        subtitle = QLabel("RTX 5060 Ti ACCELERATED VISUAL DETECTION & VIRTUAL CONTROLLER GUIDANCE")
        subtitle.setObjectName("subtle")
        brand_box.addWidget(title)
        brand_box.addWidget(subtitle)
        header.addLayout(brand_box)
        header.addStretch()
        
        self.status_pill = QLabel("● SYSTEM IDLE")
        self.status_pill.setObjectName("status")
        header.addWidget(self.status_pill)
        outer.addLayout(header)

        # Split Layout
        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, 1)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        body.addLayout(left_layout, 3)

        # Video Frame
        preview_card = self.card()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        
        preview_head = QHBoxLayout()
        preview_label = QLabel("LIVE VIDEO FEED (PS5 CAPTURE CARD)")
        preview_label.setObjectName("section")
        self.resolution_label = QLabel("NO SIGNAL")
        self.resolution_label.setObjectName("subtle")
        preview_head.addWidget(preview_label)
        preview_head.addStretch()
        preview_head.addWidget(self.resolution_label)
        preview_layout.addLayout(preview_head)

        self.preview = QLabel("Connect PS5 capture card and click 'CONNECT CAPTURE'")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(440)
        preview_layout.addWidget(self.preview, 1)
        left_layout.addWidget(preview_card, 1)

        # Joystick visualizer component
        self.joystick_widget = JoystickVisualizer(self)
        left_layout.addWidget(self.joystick_widget)

        # Controls Panel
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        body.addLayout(right_layout, 1)

        # Performance & GPU Card
        stats_card = self.card()
        sc_layout = QVBoxLayout(stats_card)
        sc_layout.setContentsMargins(14, 12, 14, 12)
        sc_layout.addWidget(self.section_label("PERFORMANCE & ACCELERATION"))
        
        grid = QGridLayout()
        grid.setSpacing(8)
        self.fps_value = self.stat_tile_in_grid(grid, 0, 0, "CAPTURE FPS", "—")
        self.infer_value = self.stat_tile_in_grid(grid, 0, 1, "GPU LATENCY", "— ms")
        self.objects_value = self.stat_tile_in_grid(grid, 1, 0, "TARGETS", "0")
        self.model_value = self.stat_tile_in_grid(grid, 1, 1, "MODEL", "NOT LOADED")
        sc_layout.addLayout(grid)

        self.gpu_device_label = QLabel("ACTIVE DEVICE: Checking...")
        self.gpu_device_label.setObjectName("gpu_status")
        sc_layout.addWidget(self.gpu_device_label)
        right_layout.addWidget(stats_card)

        # Capture Control
        capture_card = self.card()
        c_layout = QVBoxLayout(capture_card)
        c_layout.setContentsMargins(14, 12, 14, 12)
        c_layout.addWidget(self.section_label("CAPTURE CARD SETTINGS"))
        c_layout.addWidget(self.small_label("Select Video Input Device"))
        
        self.device_combo = QComboBox()
        c_layout.addWidget(self.device_combo)
        
        btn_box = QHBoxLayout()
        refresh_btn = QPushButton("REFRESH")
        refresh_btn.clicked.connect(self.refresh_devices)
        self.connect_btn = QPushButton("CONNECT CAPTURE")
        self.connect_btn.setObjectName("primary")
        self.connect_btn.clicked.connect(self.toggle_capture)
        btn_box.addWidget(refresh_btn)
        btn_box.addWidget(self.connect_btn)
        c_layout.addLayout(btn_box)
        right_layout.addWidget(capture_card)

        # AI Model & ViGEmBus Settings
        model_card = self.card()
        m_layout = QVBoxLayout(model_card)
        m_layout.setContentsMargins(14, 12, 14, 12)
        m_layout.addWidget(self.section_label("MODEL & CONTROLLER SETTINGS"))
        
        self.model_name = QLabel("No ONNX model loaded")
        self.model_name.setObjectName("subtle")
        self.model_name.setWordWrap(True)
        m_layout.addWidget(self.model_name)
        
        m_btns = QHBoxLayout()
        load_btn = QPushButton("LOAD ONNX MODEL")
        load_btn.clicked.connect(self.load_model)
        names_btn = QPushButton("CLASS NAMES")
        names_btn.clicked.connect(self.load_names)
        m_btns.addWidget(load_btn)
        m_btns.addWidget(names_btn)
        m_layout.addLayout(m_btns)

        # ViGEmBus Controller Toggle
        self.vigem_checkbox = QCheckBox("Enable ViGEmBus Virtual Controller Output")
        self.vigem_checkbox.setEnabled(VIGEM_AVAILABLE and self.gamepad is not None)
        if not VIGEM_AVAILABLE:
            self.vigem_checkbox.setToolTip("Install vgamepad & ViGEmBus drivers to enable virtual controller output.")
        self.vigem_checkbox.toggled.connect(self._on_vigem_toggled)
        m_layout.addWidget(self.vigem_checkbox)

        m_layout.addWidget(self.small_label("Confidence Threshold"))
        self.conf_label = QLabel("0.45")
        self.conf_label.setObjectName("accent")
        m_layout.addWidget(self.conf_label)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(45)
        self.conf_slider.valueChanged.connect(self._on_conf)
        m_layout.addWidget(self.conf_slider)

        m_layout.addWidget(self.small_label("NMS Box Threshold"))
        self.nms_label = QLabel("0.45")
        self.nms_label.setObjectName("accent")
        m_layout.addWidget(self.nms_label)
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(10, 90)
        self.nms_slider.setValue(45)
        self.nms_slider.valueChanged.connect(self._on_nms)
        m_layout.addWidget(self.nms_slider)
        right_layout.addWidget(model_card)

        # Pipeline Control
        run_card = self.card()
        r_layout = QVBoxLayout(run_card)
        r_layout.setContentsMargins(14, 12, 14, 12)
        r_layout.addWidget(self.section_label("VISION EXECUTION"))
        
        self.pipeline_label = QLabel("Waiting for capture & model load...")
        self.pipeline_label.setObjectName("subtle")
        self.pipeline_label.setWordWrap(True)
        r_layout.addWidget(self.pipeline_label)
        
        self.detect_btn = QPushButton("START DETECTION")
        self.detect_btn.setObjectName("accent_btn")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.toggle_detection)
        r_layout.addWidget(self.detect_btn)
        
        note = QLabel("Closest-Target Mode Enabled: Only renders and calculates vectors for the target nearest to screen center.")
        note.setObjectName("note")
        note.setWordWrap(True)
        r_layout.addWidget(note)
        right_layout.addWidget(run_card)
        right_layout.addStretch()

        # Footer
        footer = QLabel("ENGINE-X APEX VISION  •  ViGEmBus SUPPORT  •  NVIDIA CUDA ACCELERATED")
        footer.setObjectName("footer")
        outer.addWidget(footer)

    def _on_vigem_toggled(self, checked):
        self.vigem_enabled = checked
        if checked:
            print("[ViGEmBus]: Virtual controller output enabled.")
        else:
            if self.gamepad:
                self.gamepad.reset()
                self.gamepad.update()
            print("[ViGEmBus]: Virtual controller reset to neutral.")

    def _on_conf(self, v):
        self.conf_value = v / 100.0
        self.conf_label.setText(f"{v/100:.2f}")

    def _on_nms(self, v):
        self.nms_value = v / 100.0
        self.nms_label.setText(f"{v/100:.2f}")

    def card(self):
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    def section_label(self, text):
        label = QLabel(text)
        label.setObjectName("section")
        return label

    def small_label(self, text):
        label = QLabel(text)
        label.setObjectName("small")
        return label

    def stat_tile_in_grid(self, grid, r, c, name, value):
        tile = QFrame()
        tile.setObjectName("tile")
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(10, 8, 10, 8)
        n = QLabel(name)
        n.setObjectName("small")
        v = QLabel(value)
        v.setObjectName("stat")
        layout.addWidget(n)
        layout.addWidget(v)
        grid.addWidget(tile, r, c)
        return v

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0b0e14; color: #e1e4ed; font-family: 'Segoe UI', sans-serif; }
            #brand { font-size: 22px; letter-spacing: 0.5px; }
            #subtle { color: #7a8296; font-size: 11px; font-weight: 600; }
            #small { color: #8a92a6; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }
            #section { color: #76b900; font-size: 11px; font-weight: 800; letter-spacing: 1px; }
            #card { background: #121622; border: 1px solid #1e2436; border-radius: 8px; }
            #tile { background: #171d2d; border: 1px solid #232b40; border-radius: 6px; }
            #preview { background: #07090e; border: 1px solid #1a2030; border-radius: 6px; color: #5a6275; }
            #status { color: #a0a6b8; background: #161c2b; border: 1px solid #242c42; border-radius: 6px; padding: 6px 12px; font-size: 11px; font-weight: 800; }
            #stat { color: #ffffff; font-size: 20px; font-weight: 800; font-family: 'Consolas', monospace; }
            #accent { color: #76b900; font-weight: 800; }
            #gpu_status { color: #76b900; font-weight: 700; font-size: 11px; background: #152217; border: 1px solid #274720; padding: 6px; border-radius: 5px; }
            #note { color: #626a7e; font-size: 10px; font-weight: 500; }
            #footer { color: #464d5e; font-size: 10px; letter-spacing: 1px; text-align: center; }
            QPushButton { background: #1c2334; color: #d0d5e5; border: 1px solid #2b354d; border-radius: 5px; padding: 8px; font-weight: 700; font-size: 11px; }
            QPushButton:hover { background: #263047; border-color: #3b4868; }
            QPushButton#primary { background: #1f3323; color: #8ce600; border: 1px solid #365c3b; }
            QPushButton#primary:hover { background: #2a472f; }
            QPushButton#accent_btn { background: #76b900; color: #080c04; border: 1px solid #91e000; font-weight: 800; }
            QPushButton#accent_btn:hover { background: #88d400; }
            QComboBox { background: #141a28; border: 1px solid #252e45; border-radius: 5px; padding: 6px; color: #e1e4ed; font-size: 11px; }
            QCheckBox { color: #8d93a8; font-size: 11px; font-weight: 600; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            QSlider::groove:horizontal { height: 4px; background: #222a3d; border-radius: 2px; }
            QSlider::handle:horizontal { background: #76b900; width: 12px; margin: -4px 0; border-radius: 6px; }
        """)

    def _check_gpu_availability(self):
        try:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                self.gpu_device_label.setText("ACTIVE DEVICE: NVIDIA RTX GPU (CUDA Ready)")
                self.gpu_device_label.setStyleSheet("color: #76b900; background: #152217; border: 1px solid #274720; padding: 6px; border-radius: 5px;")
            else:
                self.gpu_device_label.setText("ACTIVE DEVICE: CPU (CUDA Unavailable)")
                self.gpu_device_label.setStyleSheet("color: #ff9800; background: #2b2013; border: 1px solid #543c1b; padding: 6px; border-radius: 5px;")
        except Exception as exc:
            self.gpu_device_label.setText(f"ACTIVE DEVICE: Error checking ({exc})")

    def refresh_devices(self):
        self.device_combo.clear()
        found = []
        for index in range(8):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(index)
            cap.release()
        if not found:
            self.device_combo.addItem("No capture devices found")
            self.pipeline_label.setText("No capture card detected. Check USB/HDMI connection.")
        else:
            for i in found:
                self.device_combo.addItem(f"Video Device {i} (Capture Card)", i)

    def toggle_capture(self):
        if self.capture is not None:
            self.stop_workers()
            self.capture.release()
            self.capture = None
            self.running = False
            self.latest_frame = None
            self.latest_detections = []
            self.connect_btn.setText("CONNECT CAPTURE")
            self.detect_btn.setEnabled(False)
            self.status_pill.setText("● SYSTEM IDLE")
            self.preview.setText("Capture disconnected.")
            self.resolution_label.setText("NO SIGNAL")
            self.pipeline_label.setText("Waiting for capture connection...")
            self.joystick_widget.update_target(0, 0, 0, 0, False)
            if self.gamepad and self.vigem_enabled:
                self.gamepad.reset()
                self.gamepad.update()
            return

        device = self.device_combo.currentData()
        if device is None:
            QMessageBox.warning(self, "No Device", "No capture device selected.")
            return

        cap = cv2.VideoCapture(int(device), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            QMessageBox.critical(self, "Capture Error", "Could not open video capture device.")
            return

        self.capture = cap
        self.connect_btn.setText("DISCONNECT")
        if self.session is not None:
            self.detect_btn.setEnabled(True)
        self.status_pill.setText("● CAPTURE CONNECTED")
        self.pipeline_label.setText("Capture feed live. Load model or start detection.")
        self.start_workers()
        self.timer.start(8)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Model", "", "ONNX Models (*.onnx)")
        if not path:
            return
        try:
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            
            if "CUDAExecutionProvider" in available:
                providers = [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    }),
                    "CPUExecutionProvider",
                ]
            
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(path, sess_options=opts, providers=providers)
            
            active_providers = session.get_providers()
            if "CUDAExecutionProvider" in active_providers:
                self.active_device = "NVIDIA RTX 5060 Ti (CUDA)"
                self.gpu_device_label.setText("ACTIVE DEVICE: NVIDIA RTX 5060 Ti (CUDA Accelerated)")
                self.gpu_device_label.setStyleSheet("color: #76b900; background: #152217; border: 1px solid #274720; padding: 6px; border-radius: 5px;")
            else:
                self.active_device = "CPU Fallback"
                self.gpu_device_label.setText("ACTIVE DEVICE: CPU Fallback (CUDA Not Active)")
                self.gpu_device_label.setStyleSheet("color: #ff9800; background: #2b2013; border: 1px solid #543c1b; padding: 6px; border-radius: 5px;")

            meta = session.get_inputs()[0]
            shape = meta.shape
            ih = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else INPUT_SIZE
            iw = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else INPUT_SIZE

            self.session = session
            self.input_name = meta.name
            self.input_size = (int(iw), int(ih))

            self.model_path = path
            self.model_name.setText(Path(path).name)
            self.model_value.setText("LOADED")
            if self.capture is not None:
                self.detect_btn.setEnabled(True)
            self.pipeline_label.setText(f"Loaded ONNX model successfully on {self.active_device}.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Model Load Error",
                f"Could not load ONNX model:\n\n{exc}\n\n"
                "Run CHECK_GPU.bat to verify your CUDA & NVIDIA drivers."
            )

    def load_names(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Class Names File", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            self.class_names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
            QMessageBox.information(self, "Class Names Loaded", f"Loaded {len(self.class_names)} target class labels.")
        except Exception as exc:
            QMessageBox.critical(self, "Error Loading Names", str(exc))

    def parse_detections(self, output, frame_shape):
        h, w = frame_shape[:2]
        arr = np.array(output)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return []

        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 256:
            arr = arr.T
        if arr.shape[1] < 5:
            return []

        detections = []
        conf_threshold = self.conf_value
        nms_threshold = self.nms_value
        boxes, scores, class_ids = [], [], []

        for row in arr:
            if len(row) >= 6 and len(row) > 85:
                objectness = float(row[4])
                class_scores = row[5:]
                class_id = int(np.argmax(class_scores))
                score = objectness * float(class_scores[class_id])
            else:
                class_scores = row[4:]
                if len(class_scores) == 0:
                    continue
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])

            if score < conf_threshold:
                continue

            cx, cy, bw, bh = map(float, row[:4])
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
                cx, bw = cx * self.input_size[0], bw * self.input_size[0]
                cy, bh = cy * self.input_size[1], bh * self.input_size[1]
                
            x = int((cx - bw / 2) * w / self.input_size[0])
            y = int((cy - bh / 2) * h / self.input_size[1])
            bw = int(bw * w / self.input_size[0])
            bh = int(bh * h / self.input_size[1])
            
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw <= 1 or bh <= 1:
                continue
                
            boxes.append([x, y, bw, bh])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []
            
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                detections.append((boxes[int(idx)], scores[int(idx)], class_ids[int(idx)]))
        return detections

    def start_workers(self):
        self.alive = True
        self.frame_count = 0
        self.fps = 0.0
        self.last_frame_time = time.perf_counter()
        self.cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.cap_thread.start()
        self.infer_thread.start()

    def stop_workers(self):
        self.alive = False
        self.timer.stop()
        for t in (self.cap_thread, self.infer_thread):
            if t is not None:
                t.join(timeout=2.0)
        self.cap_thread = None
        self.infer_thread = None

    def _capture_loop(self):
        while self.alive:
            cap = self.capture
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            elapsed = now - self.last_frame_time
            self.last_frame_time = now
            if elapsed > 0:
                instant = 1.0 / elapsed
                self.frame_count += 1
                self.fps = instant if self.frame_count < 3 else self.fps * 0.9 + instant * 0.1
            with self.frame_lock:
                self.latest_frame = frame
                self.latest_frame_id += 1

    def _inference_loop(self):
        last_id = -1
        while self.alive:
            if not self.running or self.session is None:
                time.sleep(0.01)
                continue
            with self.frame_lock:
                frame = self.latest_frame
                fid = self.latest_frame_id
            if frame is None or fid == last_id:
                time.sleep(0.001)
                continue
            last_id = fid
            try:
                start = time.perf_counter()
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.input_size, swapRB=True, crop=False)
                outputs = self.session.run(None, {self.input_name: blob})
                detections = self.parse_detections(outputs[0], frame.shape)
                self.last_inference_ms = (time.perf_counter() - start) * 1000.0
                self.latest_detections = detections
            except Exception as exc:
                self.latest_detections = []
                self.inference_error = str(exc)

    def update_frame(self):
        err = getattr(self, "inference_error", None)
        if err:
            self.pipeline_label.setText(f"Inference warning: {err}")
            self.inference_error = None
            
        with self.frame_lock:
            frame = self.latest_frame
            fid = self.latest_frame_id
            
        if frame is None or fid == self.shown_frame_id:
            return
            
        self.shown_frame_id = fid
        frame = frame.copy()
        h, w = frame.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0

        best_target = None
        min_dist = float('inf')

        # Draw Center Crosshair
        cv2.drawMarker(frame, (int(center_x), int(center_y)), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)

        if self.running and self.session is not None:
            detections = self.latest_detections
            
            # Find ONLY the detection closest to screen center
            for detection in detections:
                box, score, class_id = detection
                x, y, bw, bh = box
                obj_cx = x + bw / 2.0
                obj_cy = y + bh / 2.0
                
                dist = np.hypot(obj_cx - center_x, obj_cy - center_y)
                if dist < min_dist:
                    min_dist = dist
                    best_target = (detection, obj_cx - center_x, obj_cy - center_y, dist, (obj_cx, obj_cy))

        count = 1 if best_target is not None else 0

        # RENDER ONLY THE CLOSEST TARGET
        if best_target is not None:
            (box, score, class_id), dx, dy, dist, (target_x, target_y) = best_target
            x, y, bw, bh = box
            
            label = self.class_names[class_id] if class_id < len(self.class_names) else f"Target {class_id}"
            
            # EngineX Green / Accent Bounding Box ONLY for the closest target
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 230, 118), 2)
            cv2.rectangle(frame, (x, max(0, y - 22)), (x + len(label)*10 + 60, y), (0, 230, 118), -1)
            cv2.putText(frame, f"{label} {int(score*100)}% (CLOSEST)", (x + 4, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            # Visual Ray Path from Center to Target
            cv2.line(frame, (int(center_x), int(center_y)), (int(target_x), int(target_y)), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (int(target_x), int(target_y)), 5, (0, 255, 255), -1)

            # Visual overlay statistics
            cv2.putText(frame, f"CLOSEST DX: {dx:+.1f} | DY: {dy:+.1f} | DIST: {dist:.1f}px", 
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Update Joystick Visualizer with Closest Target
            self.joystick_widget.update_target(dx, dy, dist, angle_deg, True)

            # Send ViGEmBus Virtual Controller Joystick Inputs (if enabled)
            if self.gamepad and self.vigem_enabled:
                max_offset = 640.0 / 2.0
                norm_x = max(-1.0, min(1.0, dx / max_offset))
                norm_y = max(-1.0, min(1.0, dy / max_offset))
                # Map [-1.0, 1.0] to Xbox controller right joystick [-32768, 32767]
                self.gamepad.right_joystick_float(x_value_float=norm_x, y_value_float=-norm_y)
                self.gamepad.update()
        else:
            self.joystick_widget.update_target(0, 0, 0, 0, False)
            if self.gamepad and self.vigem_enabled:
                self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
                self.gamepad.update()

        self.resolution_label.setText(f"{w} × {h}")
        self.fps_value.setText(f"{self.fps:.1f}")
        self.infer_value.setText(f"{self.last_inference_ms:.1f} ms" if self.session else "— ms")
        self.objects_value.setText(str(count))
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.preview.setPixmap(pixmap)

    def toggle_detection(self):
        if self.session is None:
            QMessageBox.information(self, "Load Model", "Load a compatible ONNX model first.")
            return
        self.running = not self.running
        self.detect_btn.setText("STOP DETECTION" if self.running else "START DETECTION")
        self.status_pill.setText("● DETECTION ACTIVE" if self.running else "● CAPTURE CONNECTED")
        self.pipeline_label.setText(f"Running GPU model inference on {self.active_device}." if self.running else "Detection paused.")

    def closeEvent(self, event):
        self.stop_workers()
        if self.capture is not None:
            self.capture.release()
        if self.gamepad:
            try:
                self.gamepad.reset()
                self.gamepad.update()
            except Exception:
                pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionWindow()
    window.show()
    sys.exit(app.exec())
