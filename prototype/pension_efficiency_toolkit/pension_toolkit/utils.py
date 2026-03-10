"""Shared utilities for the pension efficiency toolkit."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_rng(seed: int = 42) -> np.random.Generator:
    """Return a deterministic NumPy default_rng with the given seed."""
    return np.random.default_rng(seed)
