"""Unit tests for exception hierarchy."""

import pytest
from rag_platform.exceptions import (
    AgentError,
    ConfigurationError,
    IngestionError,
    McpError,
    RagPlatformError,
    RetrievalError,
)


@pytest.mark.unit
def test_base_exception_formatting():
    """Verify base exception message and details formatting."""
    err_simple = RagPlatformError("Base failure")
    assert str(err_simple) == "Base failure"
    assert err_simple.details == {}

    err_detailed = RagPlatformError("Detailed failure", details={"code": 404, "path": "/doc"})
    assert "Detailed failure" in str(err_detailed)
    assert err_detailed.details["code"] == 404


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc_cls",
    [ConfigurationError, IngestionError, RetrievalError, AgentError, McpError],
)
def test_subclass_inheritance(exc_cls):
    """Verify all domain exceptions inherit from RagPlatformError."""
    instance = exc_cls("Subclass error")
    assert isinstance(instance, RagPlatformError)
    assert isinstance(instance, Exception)
