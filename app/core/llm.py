"""Centralized LLM provider configuration for FinSight AI.

Single source of truth for building chat models. The primary provider is
OpenRouter (OpenAI-compatible), with Google Gemini preserved as an optional
fallback. No API keys are ever hardcoded; everything is read from settings.

Usage:
    from app.core.llm import build_chat_model
    llm = build_chat_model(temperature=0)

Agents/tests monkeypatch their module-level ``_build_llm`` helpers, which
delegate here, so provider swaps never require touching agent logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"openrouter", "gemini"}


class LLMConfigurationError(RuntimeError):
    """Raised when the selected provider is missing required configuration."""


class LLMProviderError(RuntimeError):
    """Normalized error raised for any LLM provider failure.

    ``kind`` categorizes the failure so API layers can map it to an honest
    HTTP status instead of a generic 502.
    """

    kind = "provider_error"
    retryable = False

    def __init__(self, message: str, *, kind: str | None = None, retryable: bool | None = None):
        super().__init__(message)
        if kind is not None:
            self.kind = kind
        if retryable is not None:
            self.retryable = retryable


class LLMTimeoutError(LLMProviderError):
    kind = "timeout"
    retryable = True


class LLMRateLimitError(LLMProviderError):
    kind = "rate_limit"
    retryable = True


class LLMAuthError(LLMProviderError):
    kind = "auth"
    retryable = False


class LLMBillingError(LLMProviderError):
    kind = "billing"
    retryable = False


class LLMModelUnavailableError(LLMProviderError):
    kind = "model_unavailable"
    retryable = True


class LLMResponseError(LLMProviderError):
    kind = "invalid_response"
    retryable = False


@dataclass(frozen=True)
class LLMInfo:
    provider: str
    model: str


def get_llm_info() -> LLMInfo:
    """Return the configured provider/model (never exposes secrets)."""
    return LLMInfo(provider=settings.resolved_llm_provider, model=settings.active_llm_model)


def _openrouter_headers() -> dict[str, str]:
    headers = {"X-Title": settings.llm_app_title}
    referer = settings.llm_app_referer or settings.frontend_base_url
    if referer:
        headers["HTTP-Referer"] = referer
    return headers


def _create_openrouter_model(model: str, temperature: float) -> Any:
    if not settings.openrouter_api_key:
        raise LLMConfigurationError("OPENROUTER_API_KEY is not configured")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise LLMConfigurationError(
            "langchain-openai is required for the OpenRouter provider"
        ) from exc
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        default_headers=_openrouter_headers(),
    )


def _create_gemini_model(model: str, temperature: float) -> Any:
    if not settings.google_api_key:
        raise LLMConfigurationError("GOOGLE_API_KEY is not configured")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise LLMConfigurationError(
            "langchain-google-genai is required for the Gemini provider"
        ) from exc
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


def build_chat_model(
    *,
    temperature: float = 0.0,
    model: str | None = None,
    provider: str | None = None,
    tools: list[Any] | None = None,
) -> Any:
    """Build a LangChain chat model for the configured provider.

    The returned object is a raw chat model (no tools bound) so callers can use
    ``bind_tools`` or ``with_structured_output`` as needed. Pass ``tools`` to
    bind tools in one step.
    """
    resolved_provider = (provider or settings.resolved_llm_provider).strip().lower()
    resolved_model = model or (
        settings.gemini_model if resolved_provider == "gemini" else settings.llm_model
    )
    if resolved_provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER '{resolved_provider}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )

    logger.info(
        "Building LLM client: provider=%s model=%s temperature=%s",
        resolved_provider,
        resolved_model,
        temperature,
    )
    if resolved_provider == "gemini":
        llm = _create_gemini_model(resolved_model, temperature)
    else:
        llm = _create_openrouter_model(resolved_model, temperature)

    if tools:
        llm = llm.bind_tools(tools)
    return llm


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def classify_llm_exception(exc: Exception) -> LLMProviderError:
    """Normalize an arbitrary provider exception into an ``LLMProviderError``.

    Classification is best-effort and uses both exception type and message
    content, so it works across the OpenAI SDK, LangChain wrappers, and the
    Gemini SDK without importing provider-specific classes eagerly.
    """
    if isinstance(exc, LLMProviderError):
        return exc
    if isinstance(exc, LLMConfigurationError):
        return LLMAuthError(str(exc), kind="auth")

    status = _extract_status_code(exc)
    text = f"{type(exc).__name__}: {exc}".lower()

    rate_markers = ("rate limit", "rate_limit", "ratelimit", "too many requests", "quota", "resource_exhausted", "resource exhausted")
    auth_markers = ("invalid api key", "unauthorized", "authentication", "no auth credentials", "api key not valid", "permission")
    billing_markers = ("insufficient credits", "insufficient_quota", "spend limit", "payment required", "billing")
    unavailable_markers = ("model not found", "no endpoints found", "unknown model", "not a valid model", "no allowed providers", "overloaded", "unavailable", "temporarily")
    timeout_markers = ("timeout", "timed out", "deadline exceeded")

    if status == 429 or any(m in text for m in rate_markers):
        return LLMRateLimitError(
            "The AI provider is rate-limited right now. Please retry in a moment.",
            retryable=True,
        )
    if status in (401, 403) or any(m in text for m in auth_markers):
        return LLMAuthError(
            "The AI provider rejected the credentials (invalid or missing API key).",
            retryable=False,
        )
    if status == 402 or any(m in text for m in billing_markers):
        return LLMBillingError(
            "The AI provider account has insufficient credits or a spend limit was reached.",
            retryable=False,
        )
    if status == 404 or any(m in text for m in unavailable_markers):
        return LLMModelUnavailableError(
            "The configured AI model is currently unavailable on the provider.",
            retryable=True,
        )
    if status in (408, 504) or any(m in text for m in timeout_markers):
        return LLMTimeoutError(
            "The AI provider did not respond in time.", retryable=True
        )
    if status is not None and status >= 500:
        return LLMModelUnavailableError(
            "The AI provider is temporarily unavailable.", retryable=True
        )
    return LLMProviderError(f"AI provider error: {exc}", retryable=True)


def extract_served_model(response: Any) -> str | None:
    """Best-effort extraction of the model that actually served a response."""
    meta = getattr(response, "response_metadata", None) or {}
    if isinstance(meta, dict):
        for key in ("model_name", "model", "model_id"):
            value = meta.get(key)
            if value:
                return str(value)
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        model = usage.get("model_name") or usage.get("model")
        if model:
            return str(model)
    return None


def log_llm_response(context: str, response: Any) -> None:
    """Log which provider/model generated a response (no secrets)."""
    info = get_llm_info()
    served = extract_served_model(response)
    if served and served != info.model:
        logger.info(
            "LLM response [%s]: provider=%s requested_model=%s served_model=%s",
            context,
            info.provider,
            info.model,
            served,
        )
    else:
        logger.info(
            "LLM response [%s]: provider=%s model=%s",
            context,
            info.provider,
            info.model,
        )
