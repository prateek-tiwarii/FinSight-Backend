import base64
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from app.core.config import settings
from app.models.portfolio import PortfolioExtractionResponse


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
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or settings.google_api_key
        self.model_name = model_name or settings.gemini_model

    def _build_llm(self) -> ChatGoogleGenerativeAI:
        if not self.api_key:
            raise ExtractionError("GOOGLE_API_KEY is not configured")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0,
        )

    def _parse_json(self, content: str) -> dict[str, Any]:
        text = content.strip()
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
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ExtractionError("Model output JSON must be an object")
        return data

    async def extract(self, image_bytes: bytes, mime_type: str) -> PortfolioExtractionResponse:
        llm = self._build_llm()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        message = HumanMessage(
            content=[
                {"type": "text", "text": EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": data_url},
            ]
        )
        response = await llm.ainvoke([message])
        content = response.content if isinstance(response.content, str) else str(response.content)
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
