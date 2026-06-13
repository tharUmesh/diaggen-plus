"""
config_loader.py
Loads config.yaml and prompts.yaml from the config/ directory.
Access anywhere with: from src.utils.config_loader import cfg, prompts
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load .env on first import
load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]   # project root


def _load_yaml(filename: str) -> dict:
    path = _ROOT / "config" / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg: dict     = _load_yaml("config.yaml")
prompts: dict = _load_yaml("prompts.yaml")


def get(key_path: str, default=None):
    """Dot-notation access into cfg. E.g. get('bert.training.batch_size')"""
    keys = key_path.split(".")
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
