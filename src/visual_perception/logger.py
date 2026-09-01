"""S1 Logger Configuration.

Provides structured logging utilities for Subsystem 1 (Visual Perception).
"""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "visual_perception", log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger instance for S1.

    Parameters:
        name (str): Logger name or module name.
        log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file (Optional[str]): Path to output log file, if file logging is desired.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stream Handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler (optional)
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

