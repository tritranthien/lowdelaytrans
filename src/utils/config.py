import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_all_configs()
        return cls._instance

    def _load_all_configs(self):
        """Load all configuration files from the config directory"""
        config_dir = Path(__file__).parent.parent.parent / "config"
        
        config_files = [
            "audio_config.yaml",
            "model_config.yaml",
            "pipeline_config.yaml"
        ]

        for file_name in config_files:
            file_path = config_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if config_data:
                            self._config.update(config_data)
                            print(f"Loaded config: {file_name}")
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
            else:
                print(f"Warning: Config file not found: {file_path}")

    @classmethod
    def get(cls, key: str = None, default: Any = None) -> Any:
        """Get a configuration value by key (dot notation supported)"""
        if cls._instance is None:
            cls()
        
        if key is None:
            return cls._instance._config

        keys = key.split('.')
        value = cls._instance._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    @classmethod
    def reload(cls):
        """Reload all configurations"""
        if cls._instance:
            cls._instance._config = {}
            cls._instance._load_all_configs()

# Global accessor
def get_config(key: str = None, default: Any = None):
    return Config.get(key, default)
