"""
Configuration management utilities for the music mood classification system.
Provides centralized configuration loading and management.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: Union[str, Path] = None):
        """Initialize configuration from file."""
        if config_path is None:
            config_path = self._find_default_config()
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _find_default_config(self) -> Path:
        """Find default configuration file."""
        # Look for config in common locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path("../config/config.yaml"),
            Path("config/default.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # If no config found, create a minimal one
        logger.warning("No configuration file found, using default settings")
        return Path("config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'model': {
                'name': 'EnhancedHybridModel',
                'num_classes': 4,
                'cnn': {
                    'channels': [32, 64, 128],
                    'dropout_rate': 0.3
                },
                'mlp': {
                    'hidden_dims': [128, 64],
                    'dropout_rate': 0.2
                },
                'classifier': {
                    'hidden_dims': [256, 128],
                    'dropout_rate': 0.4
                }
            },
            'training': {
                'epochs': 50,
                'batch_size': 16,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'use_weighted_sampler': True,
                'use_class_weights': True,
                'label_smoothing': 0.1,
                'mixup_alpha': 0.2,
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 0.0001
                }
            },
            'data': {
                'processed_path': 'data/processed',
                'validation_split': 0.2,
                'test_split': 0.1,
                'audio': {
                    'sample_rate': 22050,
                    'n_mels': 128,
                    'n_mfcc': 13
                },
                'augmentation': {
                    'augment_minority_classes': True,
                    'pitch_shift_range': [-2, 2],
                    'time_stretch_range': [0.9, 1.1],
                    'noise_factor': 0.005
                }
            },
            'experiment': {
                'name': 'mood_classification',
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',
                'tensorboard': {'enabled': True},
                'wandb': {'enabled': False}
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'max_file_size': 50,
                'allowed_extensions': ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
            },
            'labels': ['happy', 'sad', 'calm', 'energetic']
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final key
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._config = deep_update(self._config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self.config_path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {path}")


# Global configuration instance
_config = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Get global configuration instance."""
    global _config
    
    if _config is None or config_path is not None:
        _config = Config(config_path)
    
    return _config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = Config(config_path)
    return _config


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file (utility function)."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file) or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    result = {}
    
    for config in configs:
        def deep_merge(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    deep_merge(d1[k], v)
                else:
                    d1[k] = v
            return d1
        
        result = deep_merge(result, config.copy())
    
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    required_keys = [
        'model.num_classes',
        'training.epochs',
        'training.batch_size',
        'training.learning_rate',
        'data.processed_path'
    ]
    
    for key in required_keys:
        keys = key.split('.')
        value = config
        
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                logger.error(f"Missing required configuration key: {key}")
                return False
            value = value[k]
    
    return True


def get_device_config() -> Dict[str, Any]:
    """Get device-specific configuration."""
    import torch
    
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }