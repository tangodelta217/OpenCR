"""
OpenCR Logging Configuration

Provides structured logging with Rich console output for development
and JSON output for production/CI environments.
"""

import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

# Global console instance
_console: Console | None = None
_configured: bool = False


def get_console() -> Console:
    """Get or create the global Rich console instance."""
    global _console
    if _console is None:
        _console = Console(stderr=True)
    return _console


def setup_logging(
    verbose: bool = False,
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: Enable DEBUG level logging.
        json_output: Use JSON format (useful for CI/production).
        log_file: Optional file path to write logs to.
    """
    global _configured

    if _configured:
        return

    # Determine log level
    level = logging.DEBUG if verbose else logging.INFO

    # Check for CI environment
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    use_json = json_output or is_ci

    # Configure handlers
    handlers: list[logging.Handler] = []

    if use_json:
        # Simple JSON-like format for CI
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        )
        handlers.append(handler)
    else:
        # Rich console handler for local development
        handlers.append(
            RichHandler(
                console=get_console(),
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
            )
        )

    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
