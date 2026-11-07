import json
import math
import socket
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget
)

APP_TITLE = "ENGINEX | CHIAKI BRIDGE"
UDP_HOST = "127.0.0.1"
UDP_PORT = 8765


class StickView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(250)
        self.dx = self.dy = self.distance = 0.0
        self.has_data = False

    def set_telemetry(self, dx, dy, distance):
        self.dx, self.dy, self.distance = dx, dy, distance
        self.has_data = True
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#111522"))
        cx, cy = self.width() / 2, self.height() / 2 - 8
        radius = min(self.width() * .30, self.height() * .36, 92)
        p.setPen(QPen(QColor("#35415b"), 2))
        p.setBrush(QBrush(QColor("#171f31")))
        p.drawEllipse(QPointF(cx, cy), radius, radius)
        p.setPen(QPen(QColor("#35415b"), 1, Qt.DotLine))
        p.drawLine(QPointF(cx-radius, cy), QPointF(cx+radius, cy))
        p.drawLine(QPointF(cx, cy-radius), QPointF(cx, cy+radius))
        p.setPen(QPen(QColor("#78849e"), 1))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(0, cy+radius+12, self.width(), 22),
                   Qt.AlignCenter, "RIGHT STICK — VISUAL TELEMETRY ONLY")
        if self.has_data:
            nx = max(-1, min(1, self.dx))
            ny = max(-1, min(1, self.dy))
            px, py = cx + nx*radius*.82, cy + ny*radius*.82
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.setBrush(QBrush(QColor("#76b900")))
            p.drawLine(QPointF(cx, cy), QPointF(px, py))
            p.drawEllipse(QPointF(px, py), 10, 10)
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor("#56617a")))
            p.drawEllipse(QPointF(cx, cy), 11, 11)


class BridgeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(850, 650)
        self.chiaki_process = None
        self.chiaki_path = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        try:
            self.sock.bind((UDP_HOST, UDP_PORT))
            self.udp_ok = True
        except OSError as exc:
            self.udp_ok = False
            self.udp_error = str(exc)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("<span style='color:white;font-size:25px;font-weight:900'>ENGINE</span><span style='color:#76b900;font-size:25px;font-weight:900'>X</span> <span style='color:#929bb0'>CHIaki BRIDGE</span>")
        header.addWidget(title)
        header.addStretch()
        self.status = QLabel("● READY")
        self.status.setStyleSheet("color:#76b900;font-weight:800")
        header.addWidget(self.status)
        layout.addLayout(header)

        subtitle = QLabel("PS5 STREAMING COMPANION  •  RTX VISION TELEMETRY  •  VISUAL-ONLY")
        subtitle.setStyleSheet("color:#7e899f;font-size:10px;font-weight:700;letter-spacing:1px")
        layout.addWidget(subtitle)

        cards = QGridLayout()
        cards.setSpacing(12)
        layout.addLayout(cards)

        chiaki = self.card()
        cl = QVBoxLayout(chiaki)
        cl.addWidget(self.heading("CHIAKI-NG"))
        self.path_label = QLabel("No Chiaki-ng executable selected")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color:#8d97ad")
        cl.addWidget(self.path_label)
        row = QHBoxLayout()
        choose = QPushButton("SELECT CHIAKI-NG")
        choose.clicked.connect(self.choose_chiaki)
        self.launch_btn = QPushButton("LAUNCH")
        self.launch_btn.setObjectName("primary")
        self.launch_btn.clicked.connect(self.launch_chiaki)
        row.addWidget(choose)
        row.addWidget(self.launch_btn)
        cl.addLayout(row)
        cards.addWidget(chiaki, 0, 0)

        telemetry = self.card()
        tl = QVBoxLayout(telemetry)
        tl.addWidget(self.heading("APEX VISION TELEMETRY"))
        self.telemetry_status = QLabel(
            f"Listening on {UDP_HOST}:{UDP_PORT}" if self.udp_ok
            else f"UDP unavailable: {self.udp_error}"
        )
        self.telemetry_status.setWordWrap(True)
        self.telemetry_status.setStyleSheet("color:#8d97ad")
        tl.addWidget(self.telemetry_status)
        self.values = QLabel("DX: —     DY: —     DISTANCE: —")
        self.values.setStyleSheet("color:#76b900;font-family:Consolas;font-weight:800")
        tl.addWidget(self.values)
        cards.addWidget(telemetry, 0, 1)

        stick_card = self.card()
        sl = QVBoxLayout(stick_card)
        sl.addWidget(self.heading("CONTROLLER VISUALIZER"))
        self.stick = StickView()
        sl.addWidget(self.stick)
        layout.addWidget(stick_card, 1)

        note = QLabel(
            "Your physical PS5 controller remains managed by Chiaki-ng. "
            "This companion displays telemetry only; it does not create or inject controller inputs."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#7e899f;font-size:11px")
        layout.addWidget(note)

        footer = QLabel("ENGINEX  •  LOCALHOST TELEMETRY  •  NO CONTROLLER INJECTION")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color:#4e5870;font-size:9px;letter-spacing:1px")
        layout.addWidget(footer)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#0b0e14;color:#e1e5ef;font-family:'Segoe UI'}
            QFrame#card{background:#121827;border:1px solid #263149;border-radius:9px}
            QLabel#heading{color:#76b900;font-size:11px;font-weight:900;letter-spacing:1px}
            QPushButton{background:#202a3e;color:#e1e5ef;border:1px solid #35415b;border-radius:6px;padding:9px;font-weight:800}
            QPushButton:hover{background:#2b3851}
            QPushButton#primary{background:#76b900;color:#081000;border-color:#91d82b}
        """)

    def card(self):
        f = QFrame()
        f.setObjectName("card")
        return f

    def heading(self, text):
        l = QLabel(text)
        l.setObjectName("heading")
        return l

    def choose_chiaki(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Chiaki-ng", "", "Applications (*.exe);;All files (*)"
        )
        if path:
            self.chiaki_path = path
            self.path_label.setText(path)

    def launch_chiaki(self):
        if self.chiaki_process and self.chiaki_process.poll() is None:
            QMessageBox.information(self, "Chiaki-ng", "Chiaki-ng is already running.")
            return
        if not self.chiaki_path or not Path(self.chiaki_path).is_file():
            QMessageBox.warning(self, "Select Chiaki-ng", "Select the Chiaki-ng executable first.")
            return
        try:
            self.chiaki_process = subprocess.Popen([self.chiaki_path])
            self.status.setText("● CHIAKI-NG LAUNCHED")
        except OSError as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))

    def poll(self):
        if self.chiaki_process and self.chiaki_process.poll() is not None:
            self.status.setText("● CHIAKI-NG CLOSED")
            self.chiaki_process = None
        if not self.udp_ok:
            return
        for _ in range(20):
            try:
                packet, _addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                data = json.loads(packet.decode("utf-8"))
                dx = max(-1.0, min(1.0, float(data["dx"])))
                dy = max(-1.0, min(1.0, float(data["dy"])))
                distance = max(0.0, float(data.get("distance", 0.0)))
            except (ValueError, TypeError, KeyError, UnicodeDecodeError):
                continue
            self.stick.set_telemetry(dx, dy, distance)
            self.values.setText(f"DX: {dx:+.2f}     DY: {dy:+.2f}     DISTANCE: {distance:.1f}")
            self.telemetry_status.setText("Receiving Apex Vision telemetry")

    def closeEvent(self, event):
        try:
            self.sock.close()
        except OSError:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BridgeWindow()
    window.show()
    sys.exit(app.exec())
