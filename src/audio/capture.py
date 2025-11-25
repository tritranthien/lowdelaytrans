import time
import numpy as np
import pyaudiowpatch as pyaudio
import sounddevice as sd
import queue
import os
import yaml
from src.utils.pipeline import ProcessBase, QueueManager
from src.utils.config import get_config

class AudioCaptureProcess(ProcessBase):
    def __init__(self):
        config = get_config("audio")
        super().__init__("AudioCapture", config)
        self.sample_rate = config.get("sample_rate", 16000)
        self.chunk_size = config.get("chunk_size", 1024)
        self.channels = config.get("channels", 1)
        self.wasapi_config = config.get("wasapi", {})
        
        # VAD settings
        self.vad_enabled = config.get("vad", {}).get("enabled", True)
        self.vad_aggressiveness = config.get("vad", {}).get("aggressiveness", 3)
        
        # Pause/Resume control
        self.paused = True  # Start paused by default
        
        # Load user settings
        self.user_settings = self._load_user_settings()
        
        # Create control queue for pause/resume commands
        self.control_queue = QueueManager.create_queue("audio_control", maxsize=10)
        self.register_input_queue("audio_control", self.control_queue)
        
        # Create output queue
        self.audio_queue = QueueManager.create_queue("audio_input", maxsize=config.get("buffer.max_queue_size", 100))
        self.register_output_queue("audio_input", self.audio_queue)
    
    def _load_user_settings(self):
        """Load user settings from config file"""
        settings_path = "config/user_settings.yaml"
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def setup(self):
        self.p = pyaudio.PyAudio()
        self.device_info = self._get_loopback_device()
        
        if not self.device_info:
            self.logger.error("No WASAPI loopback device found!")
            self.stop()
            return

        self.logger.info(f"Using loopback device: {self.device_info['name']}")
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=int(self.device_info["defaultSampleRate"]),
            input=True,
            input_device_index=self.device_info["index"],
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.logger.info("Audio capture started")

    def _get_loopback_device(self):
        """Find loopback device based on user settings or default"""
        # Check if user has selected a specific device
        audio_settings = self.user_settings.get('audio', {})
        device_index = audio_settings.get('device_index')
        
        if device_index is not None:
            try:
                device_info = self.p.get_device_info_by_index(device_index)
                self.logger.info(f"Using user-selected device: {device_info['name']}")
                return device_info
            except Exception as e:
                self.logger.warning(f"Failed to use user-selected device: {e}, falling back to default")
        
        # Default behavior: Find WASAPI loopback
        try:
            # Get default WASAPI speakers
            wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            
            self.logger.info(f"Default output device: {default_speakers['name']}")
            
            if not default_speakers["isLoopbackDevice"]:
                # Find the loopback device for the default speakers
                for loopback in self.p.get_loopback_device_info_generator():
                    # Check if names match (ignoring [Loopback] suffix if present)
                    if default_speakers["name"] in loopback["name"]:
                        self.logger.info(f"Found matching loopback device: {loopback['name']}")
                        return loopback
            else:
                return default_speakers
                
        except Exception as e:
            self.logger.error(f"Error finding loopback device: {e}")
            
        # Fallback: List all WASAPI devices to help debug
        self.logger.info("Listing all WASAPI devices:")
        try:
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                if dev["hostApi"] == pyaudio.paWASAPI:
                    self.logger.info(f"ID {i}: {dev['name']} (Loopback: {dev.get('isLoopbackDevice', False)})")
                    # Try to return the first found loopback device as a fallback
                    if dev.get("isLoopbackDevice", False):
                        self.logger.warning(f"Falling back to first available loopback device: {dev['name']}")
                        return dev
        except Exception as e:
            self.logger.error(f"Error listing devices: {e}")
            
        return None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio stream"""
        if status:
            self.logger.warning(f"Audio status: {status}")
        
        # Skip processing if paused
        if self.paused:
            return (in_data, pyaudio.paContinue)
            
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Push to queue
        try:
            self.audio_queue.put_nowait(audio_data)
            
            # Debug: Log audio level every ~100 chunks (approx 6 seconds)
            if not hasattr(self, '_chunk_count'):
                self._chunk_count = 0
            self._chunk_count += 1
            
            if self._chunk_count % 100 == 0:
                max_amp = np.max(np.abs(audio_data))
                if max_amp < 0.01:
                    self.logger.warning(f"Low audio level detected: {max_amp:.4f} (Silence?)")
                else:
                    self.logger.info(f"Audio level: {max_amp:.4f}")
                    
        except queue.Full:
            pass  # Drop frame if queue is full
            
        return (in_data, pyaudio.paContinue)

    def loop(self):
        """Main loop - check for control commands"""
        try:
            # Check for control commands (pause/resume)
            cmd = self.control_queue.get_nowait()
            if cmd == "resume":
                self.paused = False
                self.logger.info("Audio capture resumed")
            elif cmd == "pause":
                self.paused = True
                self.logger.info("Audio capture paused")
        except queue.Empty:
            pass
        
        time.sleep(0.1)

    def cleanup(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
