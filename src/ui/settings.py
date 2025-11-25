from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QComboBox, QPushButton, QMessageBox, QFormLayout)
from PySide6.QtCore import Qt
import yaml
from pathlib import Path
from src.utils.config import get_config

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        # Pass None as parent to ensure it's a top-level window and doesn't inherit transparency
        super().__init__(None) 
        self.setWindowTitle("Voice Translation Settings")
        self.setMinimumWidth(400)
        # Ensure it's visible and on top
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        # Main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        self.form_layout = QFormLayout()
        
        # Model Size Selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v2"])
        
        # Load current config
        self.config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
        self.current_config = self.load_config()
        
        current_size = self.current_config.get("asr", {}).get("whisper", {}).get("model_size", "base")
        index = self.model_combo.findText(current_size)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
            
        self.form_layout.addRow("Whisper Model Size:", self.model_combo)
        
        # Add more settings here if needed (e.g., Transparency, Font Size)
        
        self.main_layout.addLayout(self.form_layout)
        
        # Buttons
        self.button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save & Restart")
        self.cancel_btn = QPushButton("Cancel")
        
        self.save_btn.clicked.connect(self.save_settings)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.cancel_btn)
        self.main_layout.addLayout(self.button_layout)
        
    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
            return {}

    def save_settings(self):
        try:
            new_size = self.model_combo.currentText()
            
            # Update config structure
            if "asr" not in self.current_config:
                self.current_config["asr"] = {}
            if "whisper" not in self.current_config["asr"]:
                self.current_config["asr"]["whisper"] = {}
                
            self.current_config["asr"]["whisper"]["model_size"] = new_size
            
            # Save to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.current_config, f, default_flow_style=False, allow_unicode=True)
                
            QMessageBox.information(self, "Success", "Settings saved! Please restart the application to apply changes.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
