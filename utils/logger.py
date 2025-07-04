import logging
import sys

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str = ""):
    """
    Returns a logger with the specified name. If no name is provided, returns the root logger.
    """
    return logging.getLogger(name)
