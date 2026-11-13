import logging
from typing import Any

from langchain_core.tools import tool

from app.services import market_data
from app.services import news_service

logger = logging.getLogger(__name__)


def _historical_summary(hist: Any) -> dict[str, Any]:
    """Summarize historical data to avoid sending large OHLCV arrays to LLM context."""
    if hist is None:
        return {"status": "error", "message": "No historical data object"}
    status = getattr(hist, "status", None)
    summary: dict[str, Any] = {
        "symbol": getattr(hist, "symbol", None),
        "resolved_symbol": getattr(hist, "resolved_symbol", None),
        "period": getattr(hist, "period", None),
        "interval": getattr(hist, "interval", None),
        "status": status,
        "source": getattr(hist, "source", "yfinance"),
        "message": getattr(hist, "message", None),
    }
    points = getattr(hist, "points", []) or []
    if status != "ok" or not points:
        summary["observation_count"] = 0
        return summary

    closes = [p.close for p in points if getattr(p, "close", None) is not None]
    dates = [getattr(p, "date", None) for p in points if getattr(p, "date", None)]
    summary["observation_count"] = len(points)
    summary["date_start"] = dates[0] if dates else None
    summary["date_end"] = dates[-1] if dates else None
    if closes:
        summary["first_close"] = closes[0]
        summary["last_close"] = closes[-1]
        summary["min_close"] = min(closes)
        summary["max_close"] = max(closes)
        if closes[0]:
            summary["period_return_percent"] = ((closes[-1] - closes[0]) / closes[0]) * 100.0
        # small sample of recent observations (not full dataset)
        recent = points[-5:]
        summary["recent_observations"] = [
            {"date": p.date, "close": p.close, "volume": getattr(p, "volume", None)} for p in recent
        ]
    return summary


@tool("get_stock_price")
def get_stock_price(symbol: str, exchange: str | None = None) -> dict[str, Any]:
    """Get the latest available stock price for a symbol.
    Read-only. Returns latest available price (may be delayed/previous session, not live),
    currency, market timestamp, source, and data status. Use to check current valuation context."""
    try:
        if not symbol or not str(symbol).strip():
            return {"status": "error", "message": "symbol is required", "source": "yfinance"}
        res = market_data.get_stock_price(str(symbol).strip(), exchange)
        data = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        # normalize datetime
        ts = data.get("market_timestamp")
        if hasattr(ts, "isoformat"):
            data["market_timestamp"] = ts.isoformat()
        return data
    except Exception as exc:
        logger.exception("get_stock_price tool failed")
        return {"symbol": symbol, "status": "error", "message": f"Tool failure: {exc}", "source": "yfinance"}


@tool("get_historical_data")
def get_historical_data(symbol: str, period: str = "1y", exchange: str | None = None) -> dict[str, Any]:
    """Get a summarized view of historical stock performance for a symbol and period.
    Read-only. Returns performance summary (date range, closes, period return, recent observations),
    data availability, source and timestamp context - not the full OHLCV dataset.
    Use to investigate trends/volatility context when needed."""
    try:
        if not symbol or not str(symbol).strip():
            return {"status": "error", "message": "symbol is required", "source": "yfinance"}
        period = str(period or "1y")
        hist = market_data.get_historical_data(str(symbol).strip(), period=period, interval="1d", exchange=exchange)
        return _historical_summary(hist)
    except Exception as exc:
        logger.exception("get_historical_data tool failed")
        return {"symbol": symbol, "period": period, "status": "error", "message": f"Tool failure: {exc}", "source": "yfinance"}


@tool("get_company_info")
def get_company_info(symbol: str, exchange: str | None = None) -> dict[str, Any]:
    """Get available company information and fundamental metrics for a symbol.
    Read-only. Returns company name, sector, industry, available metrics (market cap, P/E, dividend yield),
    missing fields, and source. Use to understand business/sector context."""
    try:
        if not symbol or not str(symbol).strip():
            return {"status": "error", "message": "symbol is required", "source": "yfinance"}
        info = market_data.get_company_info(str(symbol).strip(), exchange)
        return info.model_dump() if hasattr(info, "model_dump") else dict(info)
    except Exception as exc:
        logger.exception("get_company_info tool failed")
        return {"symbol": symbol, "status": "error", "message": f"Tool failure: {exc}", "source": "yfinance"}


@tool("search_company_news")
def search_company_news(
    symbol: str,
    company_name: str | None = None,
    lookback_days: int = 30,
    max_articles: int = 5,
) -> dict[str, Any]:
    """Search recent news for a company/stock using Google News RSS (read-only).
    Reuses existing company information where useful. Returns compact structured articles
    (title, publisher, URL, publication date, snippet, related company/symbol) with
    explicit status/warnings and source. Use to gather evidence-backed developments."""
    try:
        symbol = (symbol or "").strip()
        company_name = (company_name or "").strip() or None
        if not symbol and not company_name:
            return {"status": "error", "message": "symbol or company_name is required", "source": "google_news_rss", "articles": []}
        # Validate/bound args
        try:
            lookback_days = max(1, min(int(lookback_days), 365))
        except Exception:
            lookback_days = 30
        try:
            max_articles = max(1, min(int(max_articles), 20))
        except Exception:
            max_articles = 5

        company_context = None
        if symbol:
            try:
                info = market_data.get_company_info(symbol)
                company_context = info.model_dump() if hasattr(info, "model_dump") else dict(info)
                if not company_name and company_context.get("company_name"):
                    company_name = company_context.get("company_name")
            except Exception:
                company_context = {"status": "error", "message": "Failed to fetch company context"}

        result = news_service.search_company_news(
            symbol=symbol,
            company_name=company_name,
            lookback_days=lookback_days,
            max_articles=max_articles,
        )
        # attach compact company context info for relevance (no duplication of large data)
        if company_context:
            result["company_context"] = {
                "company_name": company_context.get("company_name"),
                "sector": company_context.get("sector"),
                "industry": company_context.get("industry"),
                "status": company_context.get("status"),
            }
        return result
    except Exception as exc:
        logger.exception("search_company_news tool failed")
        return {
            "symbol": symbol,
            "company_name": company_name,
            "status": "error",
            "message": f"Tool failure: {exc}",
            "articles": [],
            "source": "google_news_rss",
        }


MARKET_ANALYST_TOOLS = [get_stock_price, get_historical_data, get_company_info]
RESEARCH_TOOLS = [get_company_info, search_company_news]
