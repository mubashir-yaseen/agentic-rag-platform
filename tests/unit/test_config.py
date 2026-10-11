"""Unit tests for configuration settings."""

import pytest
from rag_platform.config import Settings, get_settings
from rag_platform.exceptions import ConfigurationError


@pytest.mark.unit
def test_settings_defaults():
    """Verify default settings values."""
    settings = Settings(
        OPENAI_API_KEY="test-openai-key",
        OPENROUTER_API_KEY="",
        SUPABASE_URL="https://test.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="",
        SUPABASE_ANON_KEY="test-anon-key",
    )
    assert settings.APP_NAME == "RAG Chatbot Platform"
    assert settings.RAG_CHUNK_SIZE == 1000
    assert settings.RAG_CHUNK_OVERLAP == 200
    assert settings.effective_api_key == "test-openai-key"
    assert settings.effective_supabase_key == "test-anon-key"


@pytest.mark.unit
def test_settings_validation_raises_on_missing_api_key():
    """Verify validate_llm_credentials raises ConfigurationError when keys are missing."""
    settings = Settings(OPENAI_API_KEY="", OPENROUTER_API_KEY="")
    with pytest.raises(ConfigurationError):
        settings.validate_llm_credentials()


@pytest.mark.unit
def test_settings_supabase_key_fallback():
    """Verify SUPABASE_KEY is used as fallback when service role and anon keys are unset."""
    settings = Settings(
        SUPABASE_URL="https://test.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="",
        SUPABASE_ANON_KEY="",
        SUPABASE_KEY="test-generic-key",
    )
    assert settings.effective_supabase_key == "test-generic-key"
    # Should not raise ConfigurationError when SUPABASE_KEY is provided
    settings.validate_supabase_credentials()


@pytest.mark.unit
def test_settings_supabase_key_priority():
    """Verify priority: SUPABASE_SERVICE_ROLE_KEY > SUPABASE_ANON_KEY > SUPABASE_KEY."""
    s1 = Settings(
        SUPABASE_SERVICE_ROLE_KEY="role-key",
        SUPABASE_ANON_KEY="anon-key",
        SUPABASE_KEY="generic-key",
    )
    assert s1.effective_supabase_key == "role-key"

    s2 = Settings(
        SUPABASE_SERVICE_ROLE_KEY="",
        SUPABASE_ANON_KEY="anon-key",
        SUPABASE_KEY="generic-key",
    )
    assert s2.effective_supabase_key == "anon-key"


@pytest.mark.unit
def test_settings_validation_raises_on_missing_supabase():
    """Verify validate_supabase_credentials raises ConfigurationError when credentials are missing."""
    settings = Settings(
        SUPABASE_URL="",
        SUPABASE_SERVICE_ROLE_KEY="",
        SUPABASE_ANON_KEY="",
        SUPABASE_KEY="",
    )
    with pytest.raises(ConfigurationError):
        settings.validate_supabase_credentials()


@pytest.mark.unit
def test_get_settings_caching():
    """Verify get_settings returns the cached singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
