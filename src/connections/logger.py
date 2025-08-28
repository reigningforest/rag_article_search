"""
Logger for the application.
"""

import os
import logging
import sys
import datetime

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.INFO)

def get_shared_logger(name: str = "", dirname: str = "log", filename: str = "log") -> logging.Logger:
    """
    Returns a logger with the specified name. If no name is provided, returns the root logger.
    """

    # Create log directory if it doesn't exist
    os.makedirs(dirname, exist_ok=True)

    # Configure the file handler
    filename_now = filename + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    file_handler = logging.FileHandler(os.path.join(dirname, filename_now))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    # Ensure no duplicate file handlers
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)

    return logger
