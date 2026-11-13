"""Tests that provider errors surface as honest HTTP statuses (not a generic 502)."""

from fastapi.testclient import TestClient

from app.core.llm import (
    LLMAuthError,
    LLMBillingError,
    LLMModelUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.main import app
from app.services.extraction import ExtractionError
from tests.helpers import prime_csrf

client = TestClient(app)
prime_csrf(client)


def _post_with_error(monkeypatch, exc):
    async def boom(self, image_bytes, mime_type):
        raise exc

    monkeypatch.setattr(
        "app.services.extraction.PortfolioExtractionService.extract", boom
    )
    return client.post(
        "/api/extract/portfolio",
        files={"file": ("p.png", b"\x89PNG\r\n\x1a\nfake", "image/png")},
    )


def test_rate_limit_maps_to_429(monkeypatch):
    resp = _post_with_error(monkeypatch, LLMRateLimitError("limited"))
    assert resp.status_code == 429


def test_timeout_maps_to_504(monkeypatch):
    resp = _post_with_error(monkeypatch, LLMTimeoutError("timeout"))
    assert resp.status_code == 504


def test_auth_error_maps_to_502(monkeypatch):
    resp = _post_with_error(monkeypatch, LLMAuthError("bad key"))
    assert resp.status_code == 502


def test_billing_error_maps_to_502(monkeypatch):
    resp = _post_with_error(monkeypatch, LLMBillingError("no credits"))
    assert resp.status_code == 502


def test_model_unavailable_maps_to_503(monkeypatch):
    resp = _post_with_error(monkeypatch, LLMModelUnavailableError("down"))
    assert resp.status_code == 503


def test_extraction_error_maps_to_422(monkeypatch):
    resp = _post_with_error(monkeypatch, ExtractionError("bad output"))
    assert resp.status_code == 422
