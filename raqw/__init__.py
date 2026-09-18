"""Reach-Adaptive Quantile Window (RAQW) slope filter."""

from .config import RAQWConfig
from .filter import run_filter

__version__ = "1.0.0"

__all__ = ["RAQWConfig", "run_filter"]
