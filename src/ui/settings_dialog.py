from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                QComboBox, QRadioButton, QButtonGroup, QPushButton,
                                QGroupBox, QMessageBox)
from PySide6.QtCore import Qt, Signal
from src.audio.device_manager import list_all_devices
import yaml
import os

class SettingsDialog(QDialog):
    settings_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.setModal(True)
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Input Mode Selection
        mode_group = QGroupBox("Input Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_group = QButtonGroup()
        self.mic_radio = QRadioButton("Microphone")
        self.loopback_radio = QRadioButton("Loopback (System Audio)")
        
        self.mode_group.addButton(self.mic_radio, 0)
        self.mode_group.addButton(self.loopback_radio, 1)
        
        mode_layout.addWidget(self.mic_radio)
        mode_layout.addWidget(self.loopback_radio)
        mode_group.setLayout(mode_layout)
        
        # Device Selection
        device_group = QGroupBox("Audio Device")
        device_layout = QVBoxLayout()
        
        self.device_combo = QComboBox()
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add all to main layout
        layout.addWidget(mode_group)
        layout.addWidget(device_group)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.mic_radio.toggled.connect(self.on_mode_changed)
        self.loopback_radio.toggled.connect(self.on_mode_changed)
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button.clicked.connect(self.reject)
    
    def on_mode_changed(self):
        """Update device list when mode changes"""
        self.device_combo.clear()
        
        devices = list_all_devices()
        
        if self.mic_radio.isChecked():
            # Show microphones
            for dev in devices['input']:
                self.device_combo.addItem(dev['name'], dev)
        else:
            # Show loopback devices
            for dev in devices['loopback']:
                self.device_combo.addItem(dev['name'], dev)
    
    def load_settings(self):
        """Load settings from config file"""
        settings_path = "config/user_settings.yaml"
        
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f) or {}
        else:
            settings = {}
        
        # Set mode
        mode = settings.get('audio', {}).get('input_mode', 'loopback')
        if mode == 'microphone':
            self.mic_radio.setChecked(True)
        else:
            self.loopback_radio.setChecked(True)
        
        # Trigger device list update
        self.on_mode_changed()
        
        # Select saved device
        device_name = settings.get('audio', {}).get('device_name')
        if device_name:
            index = self.device_combo.findText(device_name)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
    
    def save_settings(self):
        """Save settings to config file"""
        if self.device_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Warning", "Please select a device")
            return
        
        mode = 'microphone' if self.mic_radio.isChecked() else 'loopback'
        device_name = self.device_combo.currentText()
        device_data = self.device_combo.currentData()
        
        settings = {
            'audio': {
                'input_mode': mode,
                'device_name': device_name,
                'device_index': device_data['index']
            }
        }
        
        # Save to file
        settings_path = "config/user_settings.yaml"
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        
        with open(settings_path, 'w', encoding='utf-8') as f:
            yaml.dump(settings, f, default_flow_style=False)
        
        # Emit signal
        self.settings_changed.emit(settings)
        
        QMessageBox.information(self, "Success", "Settings saved! Please restart the application for changes to take effect.")
        self.accept()
