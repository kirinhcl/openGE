"""Configuration parser for model and training parameters."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager supporting YAML and JSON formats.
    
    Handles model architecture, training hyperparameters, and crop-specific settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config file (YAML or JSON)
        """
        self.config = {}
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file
        """
        path = Path(config_path)
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path) as f:
                self.config = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def save(self, config_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save config
        """
        path = Path(config_path)
        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.dump(self.config, f)
            elif path.suffix == ".json":
                json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set config value.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            value: Value to set
        """
        keys = key.split(".")
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self.config
