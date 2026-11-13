"""Tests for the centralized LLM provider configuration (OpenRouter + Gemini)."""

import pytest

from app.core.config import settings
from app.core.llm import (
    LLMAuthError,
    LLMConfigurationError,
    LLMModelUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
    build_chat_model,
    classify_llm_exception,
    extract_served_model,
    get_llm_info,
)


class FakeHTTPError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code} error")
        self.status_code = status_code


def test_get_llm_info_reports_configured_provider():
    info = get_llm_info()
    assert info.provider in {"openrouter", "gemini"}
    assert info.model


def test_build_chat_model_openrouter(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    model = build_chat_model(temperature=0)
    assert type(model).__name__ == "ChatOpenAI"


def test_build_chat_model_gemini(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    model = build_chat_model(provider="gemini", temperature=0)
    assert type(model).__name__ == "ChatGoogleGenerativeAI"


def test_build_chat_model_unsupported_provider():
    with pytest.raises(LLMConfigurationError):
        build_chat_model(provider="not-a-provider")


def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    with pytest.raises(LLMConfigurationError):
        build_chat_model(provider="openrouter")


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    with pytest.raises(LLMConfigurationError):
        build_chat_model(provider="gemini")


def test_build_chat_model_binds_tools(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")

    def sample_tool(x: str) -> str:
        return x

    bound = build_chat_model(tools=[sample_tool])
    assert bound is not None


@pytest.mark.parametrize(
    "exc,expected",
    [
        (FakeHTTPError(429), LLMRateLimitError),
        (Exception("rate limit exceeded"), LLMRateLimitError),
        (Exception("RESOURCE_EXHAUSTED quota"), LLMRateLimitError),
        (FakeHTTPError(401), LLMAuthError),
        (Exception("Invalid API key provided"), LLMAuthError),
        (FakeHTTPError(404), LLMModelUnavailableError),
        (Exception("model not found"), LLMModelUnavailableError),
        (FakeHTTPError(408), LLMTimeoutError),
        (Exception("Request timed out"), LLMTimeoutError),
    ],
)
def test_classify_llm_exception(exc, expected):
    classified = classify_llm_exception(exc)
    assert isinstance(classified, expected)


def test_classify_is_idempotent():
    original = LLMRateLimitError("limited")
    assert classify_llm_exception(original) is original


def test_extract_served_model():
    class Resp:
        response_metadata = {"model_name": "nvidia/nemotron-3.5-super-vl:free"}

    assert extract_served_model(Resp()) == "nvidia/nemotron-3.5-super-vl:free"


def test_extract_served_model_missing():
    class Resp:
        response_metadata = {}

    assert extract_served_model(Resp()) is None
