import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = "sentiment_analyser", log_file: str = "logs/app.log", level: str = "INFO") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Windows PowerShell / cmd may not support UTF-8 by default.
    # Use errors='replace' so Unicode chars never crash the console handler.
    console_stream = sys.stdout
    try:
        console_stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)
    except Exception:
        pass  # fallback to default stdout

    console_handler = logging.StreamHandler(console_stream)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
