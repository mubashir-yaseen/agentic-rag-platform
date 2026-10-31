"""Shared test setup: the unit tests must pass on a clean machine (and in CI) with no real secrets."""

import os

# Dummy values so settings that "require" credentials exist. Real services are always mocked in unit tests.
# setdefault never overrides a value that is already set in the environment.
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("SUPABASE_URL", "https://test-project.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
