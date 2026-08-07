"""Reach-Adaptive Quantile Window (RAQW) slope filter."""

from .config import RAQWConfig
from .filter import run_filter

__version__ = "0.1.0"

__all__ = ["RAQWConfig", "run_filter"]
