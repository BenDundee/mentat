"""Logging configuration for Mentat."""

import logging
import os
from pathlib import Path

LOG_DIR = Path("log")
LOG_FILE = LOG_DIR / "mentat.log"

_configured = False


def setup_logging() -> None:
    """Configure logging to file and console."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(exist_ok=True)

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call setup_logging() first."""
    return logging.getLogger(name)
