"""Structured logging configuration with correlation ID tracking."""

import contextvars
import json
import logging
import sys
import uuid
from typing import Any, Optional

# Context variable to hold correlation ID per async task or thread
_correlation_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> str:
    """Retrieve the current correlation ID, generating a new one if unset."""
    corr_id = _correlation_id_ctx.get()
    if not corr_id:
        corr_id = str(uuid.uuid4())
        _correlation_id_ctx.set(corr_id)
    return corr_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id_ctx.set(correlation_id)


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", _correlation_id_ctx.get() or "none"),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Include custom extra attributes if passed
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "correlation_id", "message"
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        if extras:
            log_record["extra"] = extras

        return json.dumps(log_record)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that injects the current correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id") or not record.correlation_id:
            record.correlation_id = _correlation_id_ctx.get() or "none"
        return True


def setup_logging(level: int = logging.INFO, json_format: bool = True) -> None:
    """Configure root logger with structured JSON or readable console output."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(CorrelationIdFilter())

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(correlation_id)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Obtain a logger instance with the CorrelationIdFilter attached."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, CorrelationIdFilter) for f in logger.filters):
        logger.addFilter(CorrelationIdFilter())
    return logger
