"""Tests for OpenRouter/Nemotron-based screenshot extraction."""

import asyncio
import json

import pytest

from app.core.llm import LLMRateLimitError
from app.services.extraction import ExtractionError, PortfolioExtractionService


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {"model_name": "nvidia/nemotron-3.5-super-vl:free"}


class FakeAsyncLLM:
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        if self._exc is not None:
            raise self._exc
        return FakeResponse(self._content)


def _run(service, image=b"\x89PNG\r\n\x1a\nfake", mime="image/png"):
    return asyncio.run(service.extract(image, mime))


def test_extraction_parses_valid_json(monkeypatch):
    payload = {
        "holdings": [
            {"stock_symbol": "INFY", "quantity": 5, "average_purchase_price": 1500.0}
        ],
        "warnings": [],
        "extraction_notes": "ok",
    }
    fake = FakeAsyncLLM(content=json.dumps(payload))
    monkeypatch.setattr(PortfolioExtractionService, "_build_llm", lambda self: fake)

    result = _run(PortfolioExtractionService())
    assert result.requires_confirmation is True
    assert result.source == "screenshot"
    assert result.holdings[0].stock_symbol == "INFY"
    assert result.holdings[0].quantity == 5
    assert result.holdings[0].average_purchase_price == 1500.0


def test_extraction_uses_openai_multimodal_format(monkeypatch):
    fake = FakeAsyncLLM(content=json.dumps({"holdings": [], "warnings": []}))
    monkeypatch.setattr(PortfolioExtractionService, "_build_llm", lambda self: fake)

    _run(PortfolioExtractionService(provider="openrouter"))
    content = fake.messages[0].content
    assert content[0]["type"] == "text"
    image_block = content[1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


def test_extraction_strips_markdown_fences(monkeypatch):
    payload = {"holdings": [{"stock_symbol": "TCS", "quantity": 1, "average_purchase_price": 10.0}]}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    fake = FakeAsyncLLM(content=fenced)
    monkeypatch.setattr(PortfolioExtractionService, "_build_llm", lambda self: fake)

    result = _run(PortfolioExtractionService())
    assert result.holdings[0].stock_symbol == "TCS"


def test_extraction_malformed_json_raises(monkeypatch):
    fake = FakeAsyncLLM(content="this is not json at all")
    monkeypatch.setattr(PortfolioExtractionService, "_build_llm", lambda self: fake)
    with pytest.raises(ExtractionError):
        _run(PortfolioExtractionService())


def test_extraction_empty_response_raises(monkeypatch):
    fake = FakeAsyncLLM(content="")
    monkeypatch.setattr(PortfolioExtractionService, "_build_llm", lambda self: fake)
    with pytest.raises(ExtractionError):
        _run(PortfolioExtractionService())


def test_extraction_invalid_schema_raises(monkeypatch):
    fake = FakeAsyncLLM(content=json.dumps({"holdings": [{"stock_symbol": "X", "quantity": "abc"}]}))
    monkeypatch.setattr(PortfolioExtractionService, "_build_llm", lambda self: fake)
    with pytest.raises(ExtractionError):
        _run(PortfolioExtractionService())


def test_extraction_provider_rate_limit_is_classified(monkeypatch):
    fake = FakeAsyncLLM(exc=Exception("429 rate limit exceeded"))
    monkeypatch.setattr(PortfolioExtractionService, "_build_llm", lambda self: fake)
    with pytest.raises(LLMRateLimitError):
        _run(PortfolioExtractionService())


def test_image_content_format_differs_by_provider():
    gemini = PortfolioExtractionService(provider="gemini")._image_content("data:x")
    assert isinstance(gemini["image_url"], str)
    openrouter = PortfolioExtractionService(provider="openrouter")._image_content("data:x")
    assert isinstance(openrouter["image_url"], dict)
    assert openrouter["image_url"]["url"] == "data:x"
