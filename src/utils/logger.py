"""
logger.py
Centralised logging setup. Call get_logger(__name__) in every module.
"""

import logging
import sys
from pathlib import Path
from src.utils.config_loader import get

_LOG_DIR = Path("reports/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LEVEL = getattr(logging, get("logging.level", "INFO").upper(), logging.INFO)

logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "diaggen.log", encoding="utf-8"),
    ],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
