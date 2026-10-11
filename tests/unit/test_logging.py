"""Unit tests for structured logging and correlation IDs."""

import json
import logging
import pytest
from rag_platform.logging_config import (
    JSONFormatter,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)


@pytest.mark.unit
def test_correlation_id_lifecycle():
    """Verify correlation ID retrieval, setting, and custom values."""
    custom_id = "req-12345-abcde"
    set_correlation_id(custom_id)
    assert get_correlation_id() == custom_id


@pytest.mark.unit
def test_json_formatter_output():
    """Verify JSONFormatter produces valid JSON with expected fields."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Structured log message",
        args=(),
        exc_info=None,
    )
    record.correlation_id = "test-corr-id"
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Structured log message"
    assert parsed["correlation_id"] == "test-corr-id"
    assert "timestamp" in parsed


@pytest.mark.unit
def test_get_logger():
    """Verify get_logger returns a properly configured logger."""
    logger = get_logger("my_module")
    assert logger.name == "my_module"
