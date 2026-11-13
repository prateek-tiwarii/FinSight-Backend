import base64
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.core.config import settings
from app.core.llm import (
    LLMProviderError,
    build_chat_model,
    classify_llm_exception,
    get_llm_info,
    log_llm_response,
)
from app.models.portfolio import PortfolioExtractionResponse

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """
You are a portfolio screenshot parser for Groww portfolio screenshots.
Extract stock holdings from the image and return ONLY valid JSON matching this schema:
{
  "holdings": [
    {
      "stock_symbol": "string",
      "quantity": number,
      "average_purchase_price": number
    }
  ],
  "warnings": ["string"],
  "extraction_notes": "string or null"
}

Rules:
- Extract only symbols/stocks, quantity, and average purchase price visible in the image.
- Do not invent missing or unreadable values. If a field is unreadable for a holding, omit that holding and add a warning.
- Use numbers only for quantity and average_purchase_price (no currency symbols, commas).
- If no holdings can be confidently extracted, return empty holdings and explain in warnings/notes.
- Return JSON only. No markdown, no commentary.
""".strip()


class ExtractionError(Exception):
    pass


class PortfolioExtractionService:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        provider: str | None = None,
    ):
        # ``api_key`` is accepted for backward compatibility but keys always come
        # from settings; only ``model_name``/``provider`` overrides are honored.
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider

    def _build_llm(self):
        try:
            return build_chat_model(
                temperature=0, model=self.model_name, provider=self.provider
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise ExtractionError(str(exc)) from exc

    def _image_content(self, data_url: str) -> dict[str, Any]:
        """Return provider-correct multimodal image content block."""
        provider = (self.provider or settings.resolved_llm_provider).strip().lower()
        if provider == "gemini":
            # langchain-google-genai accepts a string data URL for image_url.
            return {"type": "image_url", "image_url": data_url}
        # OpenAI-compatible (OpenRouter) expects an object with a ``url`` key.
        return {"type": "image_url", "image_url": {"url": data_url}}

    def _parse_json(self, content: str) -> dict[str, Any]:
        text = (content or "").strip()
        # Strip markdown fences if model includes them despite instructions
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Try to recover JSON object from surrounding text
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ExtractionError(f"Model returned non-JSON output: {text[:200]}") from exc
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as exc2:
                raise ExtractionError(
                    f"Model returned malformed JSON: {text[:200]}"
                ) from exc2
        if not isinstance(data, dict):
            raise ExtractionError("Model output JSON must be an object")
        return data

    async def extract(self, image_bytes: bytes, mime_type: str) -> PortfolioExtractionResponse:
        if not image_bytes:
            raise ExtractionError("Empty image provided for extraction.")
        llm = self._build_llm()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        message = HumanMessage(
            content=[
                {"type": "text", "text": EXTRACTION_PROMPT},
                self._image_content(data_url),
            ]
        )
        try:
            response = await llm.ainvoke([message])
        except Exception as exc:
            provider_error = classify_llm_exception(exc)
            logger.warning(
                "Portfolio extraction LLM call failed (provider=%s model=%s): %s",
                get_llm_info().provider,
                get_llm_info().model,
                provider_error,
            )
            raise provider_error from exc

        log_llm_response("portfolio_extraction", response)

        raw_content = response.content
        if isinstance(raw_content, list):
            # Some providers return content blocks; join text parts.
            content = "".join(
                part.get("text", "")
                for part in raw_content
                if isinstance(part, dict)
            )
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            content = str(raw_content)

        if not content.strip():
            raise ExtractionError("Model returned an empty response.")

        data = self._parse_json(content)

        # Normalize and validate
        payload = {
            "holdings": data.get("holdings", []),
            "requires_confirmation": True,
            "source": "screenshot",
            "warnings": data.get("warnings", []) or [],
            "extraction_notes": data.get("extraction_notes"),
        }
        try:
            validated = PortfolioExtractionResponse.model_validate(payload)
        except ValidationError as exc:
            raise ExtractionError(f"Extracted data failed validation: {exc}") from exc
        return validated
