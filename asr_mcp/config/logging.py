import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def _parse_log_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def setup_logging(log_dir: Path = Path("./logs"), log_level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    level = _parse_log_level(log_level)

    logger = logging.getLogger("asr_mcp")
    logger.setLevel(level)
    logger.handlers.clear()

    file_handler = RotatingFileHandler(log_file, maxBytes=10_485_760, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "asr_mcp") -> logging.Logger:
    return logging.getLogger(name)
