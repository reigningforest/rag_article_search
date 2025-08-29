"""Connection functions."""

from .logger import get_shared_logger
from .gemini_query import (
    setup_client,
    query_client
)

__all__ = [
    "get_shared_logger",
    "setup_client",
    "query_client"
]