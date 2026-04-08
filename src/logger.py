"""Shared logger helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from src.runtime_paths import ensure_runtime_dirs, logs_dir


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)

    try:
        ensure_runtime_dirs()
        file_handler = logging.FileHandler(
            Path(logs_dir()) / "voiceworkbench.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)
    except Exception:
        # Logging should never block the app from starting.
        pass

    logger.propagate = False
    return logger
