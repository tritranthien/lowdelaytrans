import sys
import queue
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, 
                                QSystemTrayIcon, QMenu)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QAction, QIcon, QPixmap, QPainter, QColor
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config
from src.ui.settings_dialog import SettingsDialog
import multiprocessing as mp

class OverlayWorker(QThread):
    text_received = Signal(str)

    def __init__(self, input_queue):
        super().__init__()
        self.input_queue = input_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                text = self.input_queue.get(timeout=0.1)
                if text:
                    self.text_received.emit(text)
            except queue.Empty:
                continue
            except Exception:
                continue

    def stop(self):
        self.running = False
        self.wait()

class SubtitleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = get_config("ui.overlay")
        
        # Window setup
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.WindowTransparentForInput  # Click-through
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Layout - Horizontal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setAlignment(Qt.AlignBottom | Qt.AlignCenter)
        
        # Single label for horizontal scrolling text
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(False)  # No wrap for horizontal display
        
        font_config = self.config.get("font", {})
        font = QFont(font_config.get("family", "Arial"), font_config.get("size", 24))
        font.setBold(True)
        self.label.setFont(font)
        
        self.label.setStyleSheet(f"""
            QLabel {{
                color: {font_config.get("color", "#FFFFFF")};
                background-color: rgba(0, 0, 0, 150);
                padding: 15px 30px;
                border-radius: 10px;
            }}
        """)
        
        self.layout.addWidget(self.label)
        
        # Position - wider for horizontal text
        screen = QApplication.primaryScreen().geometry()
        width = self.config.get("size", [1200, 100])[0]  # Wider
        height = self.config.get("size", [1200, 100])[1]  # Shorter
        self.setGeometry(
            (screen.width() - width) // 2,
            screen.height() - height - 50,
            width,
            height
        )
        
        # Text buffer for horizontal scrolling
        self.text_buffer = []
        self.max_sentences = 2  # Show last 2 sentences
        
        # Timer to clear old text
        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(self.clear_oldest_sentence)
        self.sentence_timeout = self.config.get("display_duration", 8.0) * 1000

    def update_text(self, text):
        """Add new translation to horizontal display"""
        # Add to buffer
        self.text_buffer.append(text)
        
        # Keep only last N sentences
        if len(self.text_buffer) > self.max_sentences:
            self.text_buffer.pop(0)
        
        # Join with separator for horizontal display
        display_text = "  |  ".join(self.text_buffer)
        self.label.setText(display_text)
        self.show()
        
        # Restart timer
        self.clear_timer.start(int(self.sentence_timeout))

    def clear_oldest_sentence(self):
        """Remove the oldest sentence"""
        if self.text_buffer:
            self.text_buffer.pop(0)
            
            if self.text_buffer:
                display_text = "  |  ".join(self.text_buffer)
                self.label.setText(display_text)
            else:
                self.label.setText("")
                self.clear_timer.stop()

class OverlayProcess(ProcessBase):
    def __init__(self):
        config = get_config("ui")
        super().__init__("OverlayUI", config)
        self.input_queue = QueueManager.get_queue("ui_input")
        self.register_input_queue("ui_input", self.input_queue)
        self.is_running = False  # Start/Stop state
        
        # Control queue to send commands to AudioCapture
        self.audio_control_queue = QueueManager.get_queue("audio_control")
        self.register_output_queue("audio_control", self.audio_control_queue)

    def loop(self):
        pass

    def _create_tray_icon(self):
        """Create a simple colored icon programmatically"""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw green circle background
        painter.setBrush(QColor("#00FF00"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        
        # Draw "T" text
        painter.setPen(QColor("#000000"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(40)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "T")
        
        painter.end()
        return QIcon(pixmap)

    def run(self):
        self.logger.info(f"Process {self.name} started")
        
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False) # Keep app running for tray icon
        
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.logger.error("System Tray is not available on this system!")
        
        # Create transparent overlay window
        window = SubtitleWindow()
        window.show()
        
        # Setup System Tray Icon
        tray_icon = QSystemTrayIcon(app)
        tray_icon.setIcon(self._create_tray_icon())
        tray_icon.setToolTip("Voice Translator")
        
        # Tray Menu
        menu = QMenu()
        
        # Start/Stop action
        self.start_stop_action = QAction("▶ Start", parent=menu)
        self.start_stop_action.triggered.connect(lambda: self.toggle_start_stop(tray_icon))
        menu.addAction(self.start_stop_action)
        
        menu.addSeparator()
        
        settings_action = QAction("Settings", parent=menu)
        settings_action.triggered.connect(lambda: self.open_settings(window))
        menu.addAction(settings_action)
        
        menu.addSeparator()
        
        exit_action = QAction("Exit", parent=menu)
        exit_action.triggered.connect(app.quit)
        menu.addAction(exit_action)
        
        tray_icon.setContextMenu(menu)
        tray_icon.show()
        self.logger.info("System Tray Icon created and shown")
        
        # Worker thread
        worker = OverlayWorker(self.input_queue)
        worker.text_received.connect(window.update_text)
        worker.start()
        
        sys.exit(app.exec())

    def toggle_start_stop(self, tray_icon):
        """Toggle Start/Stop state"""
        self.is_running = not self.is_running
        
        if self.is_running:
            self.start_stop_action.setText("⏸ Stop")
            tray_icon.setToolTip("Voice Translator (Running)")
            self.logger.info("Started capturing audio")
            # Send resume command to AudioCapture
            try:
                self.audio_control_queue.put_nowait("resume")
            except:
                pass
        else:
            self.start_stop_action.setText("▶ Start")
            tray_icon.setToolTip("Voice Translator (Stopped)")
            self.logger.info("Stopped capturing audio")
            # Send pause command to AudioCapture
            try:
                self.audio_control_queue.put_nowait("pause")
            except:
                pass
    
    def open_settings(self, parent):
        dialog = SettingsDialog(parent)
        dialog.exec()
