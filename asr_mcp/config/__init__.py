import json
from pathlib import Path
from typing import Any

_config: dict | None = None
_DEBUG: bool = False

_CONFIG_PATH = Path(__file__).parent / "thresholds.json"


def get_config() -> dict:
    global _config
    if _config is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                _config = json.load(f)
        else:
            _config = {}
    return _config


def get(section: str, key: str = None, default: Any = None) -> Any:
    cfg = get_config()
    section_data = cfg.get(section, {})
    if key is None:
        return section_data if section_data else default
    return section_data.get(key, default)


def is_debug() -> bool:
    return _DEBUG or get("debug", default=False)


def set_debug(enabled: bool):
    global _DEBUG
    _DEBUG = enabled
