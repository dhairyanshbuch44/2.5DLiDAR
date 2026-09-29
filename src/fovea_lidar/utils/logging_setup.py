"""Centralized logging setup for Fovea-LiDAR."""

from __future__ import annotations

import logging
import sys
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGGING_INITIALIZED = False


def setup_logging(
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """Configure centralized structured logging for the Fovea-LiDAR system.

    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).
    log_format : str, optional
        Custom log message format.
    date_format : str, optional
        Custom timestamp format.
    stream : file-like, optional
        Stream to write logs to (default: sys.stdout).
    """
    global _LOGGING_INITIALIZED

    fmt = log_format or DEFAULT_LOG_FORMAT
    datefmt = date_format or DEFAULT_DATE_FORMAT
    output_stream = stream or sys.stdout

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(output_stream)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    _LOGGING_INITIALIZED = True


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """Return a logger instance, initializing centralized logging if not already done.

    Parameters
    ----------
    name : str, optional
        Logger name (typically __name__).
    level : int, optional
        Optional log level override.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if not _LOGGING_INITIALIZED:
        setup_logging()

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
