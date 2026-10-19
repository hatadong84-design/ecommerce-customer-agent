import logging
import os
import datetime

from 智能体.path_toll import get_abs_path

LOG_ROOT = get_abs_path("智能体/logs")

os.makedirs(LOG_ROOT, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if log_file is None:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()

if __name__ == "__main__":
    logger.info("logger initialized")
    logger.error("this is an error message")
    logger.warning("this is a warning message")
    logger.debug("this is a debug message")

