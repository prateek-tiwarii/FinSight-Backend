"""Manual integration test against the real OpenRouter API.

This is NOT collected by pytest (filename does not start with ``test_``).
Run explicitly from the ``backend`` directory:

    .venv\\Scripts\\python.exe -m tests.manual_openrouter_integration

It exercises the actual provider: model init, text completion, tool calling,
and vision-based screenshot extraction. Requires a valid OPENROUTER_API_KEY in
the backend environment (never hardcode keys).
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from app.core.config import settings
from app.core.llm import (
    build_chat_model,
    classify_llm_exception,
    extract_served_model,
    get_llm_info,
)
from app.services.extraction import PortfolioExtractionService

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def _sample_portfolio_png() -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (720, 240), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "Groww Portfolio Holdings",
        "Stock          Qty    Avg Price",
        "INFY           5      1500.00",
        "TCS            10     3400.50",
        "RELIANCE       2      2450.00",
    ]
    y = 20
    for line in lines:
        draw.text((30, y), line, fill="black")
        y += 34
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_model_init() -> None:
    info = get_llm_info()
    try:
        model = build_chat_model(temperature=0)
        record("model init", True, f"provider={info.provider} model={info.model} -> {type(model).__name__}")
    except Exception as exc:  # noqa: BLE001
        record("model init", False, str(exc))


def test_text_completion() -> None:
    try:
        model = build_chat_model(temperature=0)
        resp = model.invoke("Reply with exactly the word: OK")
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        served = extract_served_model(resp)
        meta_keys = list((getattr(resp, "response_metadata", None) or {}).keys())
        record(
            "text completion",
            bool(content.strip()),
            f"served={served} meta_keys={meta_keys} content={content.strip()[:60]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        record("text completion", False, f"{type(exc).__name__}: {exc}")


def test_tool_calling() -> None:
    from langchain_core.messages import HumanMessage, ToolMessage

    def get_price(symbol: str) -> dict[str, Any]:
        """Get the latest stock price for a symbol."""
        return {"symbol": symbol, "price": 123.45}

    try:
        model = build_chat_model(temperature=0).bind_tools([get_price])
        first = model.invoke([HumanMessage(content="What is the current price of INFY? Use the tool.")])
        tool_calls = getattr(first, "tool_calls", None) or []
        if not tool_calls:
            record("tool calling", False, f"no tool calls returned: {first.content!r}")
            return
        # Execute each tool call and feed matching ToolMessages back (round-trip).
        messages = [
            HumanMessage(content="What is the current price of INFY? Use the tool."),
            first,
        ]
        for call in tool_calls:
            result = get_price(**call.get("args", {}))
            messages.append(
                ToolMessage(content=json.dumps(result), tool_call_id=call.get("id"))
            )
        second = model.invoke(messages)
        final_text = second.content if isinstance(second.content, str) else str(second.content)
        record(
            "tool calling round-trip",
            len(tool_calls) > 0 and bool(final_text.strip()),
            f"tool_calls={[c.get('name') for c in tool_calls]} final={final_text.strip()[:80]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        record("tool calling round-trip", False, f"{type(exc).__name__}: {exc}")


def test_structured_output() -> None:
    from app.models.agent import MarketAnalystFindings

    try:
        model = build_chat_model(temperature=0)
        structured = model.with_structured_output(MarketAnalystFindings)
        result = structured.invoke(
            "Return a MarketAnalystFindings JSON with portfolio_overview='test' and one key finding."
        )
        record(
            "structured output",
            isinstance(result, MarketAnalystFindings) and bool(result.portfolio_overview),
            f"type={type(result).__name__} overview={getattr(result, 'portfolio_overview', None)!r}",
        )
    except Exception as exc:  # noqa: BLE001
        record("structured output", False, f"{type(exc).__name__}: {exc}")


def test_vision_extraction() -> None:
    try:
        png = _sample_portfolio_png()
        service = PortfolioExtractionService()
        result = asyncio.run(service.extract(png, "image/png"))
        symbols = [h.stock_symbol for h in result.holdings]
        record(
            "vision extraction",
            len(result.holdings) > 0,
            f"holdings={len(result.holdings)} symbols={symbols} warnings={result.warnings}",
        )
        print("    raw result:", result.model_dump_json())
    except Exception as exc:  # noqa: BLE001
        classified = classify_llm_exception(exc)
        record("vision extraction", False, f"{type(classified).__name__}: {classified}")


def main() -> int:
    if not settings.openrouter_api_key and settings.resolved_llm_provider == "openrouter":
        print("OPENROUTER_API_KEY is not configured; aborting manual integration test.")
        return 2
    info = get_llm_info()
    print(f"Running manual OpenRouter integration test (provider={info.provider}, model={info.model})\n")
    test_model_init()
    test_text_completion()
    test_tool_calling()
    test_structured_output()
    test_vision_extraction()
    print("\nSummary:")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(json.dumps([{"test": n, "ok": ok, "detail": d} for n, ok, d in RESULTS], indent=2))
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
