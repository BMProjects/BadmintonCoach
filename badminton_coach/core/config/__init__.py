"""Configuration: pydantic schema + YAML loader."""

from .loader import load_config
from .schema import AppConfig, IOConfig, ModuleConfig, PerceptionConfig

__all__ = ["load_config", "AppConfig", "IOConfig", "ModuleConfig", "PerceptionConfig"]
