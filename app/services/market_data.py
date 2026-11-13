import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from app.models.analysis import CompanyInfo, HistoricalData, MarketPrice, OHLCVPoint

logger = logging.getLogger(__name__)

# Unambiguous common-name/ticker shortcuts for Indian equities (Groww context).
# Ambiguous names must NOT be auto-mapped.
COMMON_SYMBOL_MAP: dict[str, str] = {
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "LARSEN & TOUBRO": "LT.NS",
    "LARSEN AND TOUBRO": "LT.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "HUL": "HINDUNILVR.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "KOTAK MAHINDRA BANK": "KOTAKBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "HCL TECHNOLOGIES": "HCLTECH.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "POWERGRID": "POWERGRID.NS",
    "TITAN": "TITAN.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    # Bank / financial names commonly written with abbreviations
    "SBI": "SBIN.NS",
    "SBI BANK": "SBIN.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    # Power / utilities
    "TATAPOWER": "TATAPOWER.NS",
    "TATA POWER": "TATAPOWER.NS",
    "JPPOWER": "JPPOWER.NS",
    "JP POWER": "JPPOWER.NS",
    "JAIPRAKASHPOWER": "JPPOWER.NS",
    "JAIPRAKASH POWER": "JPPOWER.NS",
    "JAIPRAKASH POWER VENTURES": "JPPOWER.NS",
    "ADANIGREEN": "ADANIGREEN.NS",
    "ADANI GREEN": "ADANIGREEN.NS",
    "ADANI GREEN ENERGY": "ADANIGREEN.NS",
    "IREDA": "IREDA.NS",
    "INDIAN RENEWABLE ENERGY": "IREDA.NS",
    "INDIAN REN ENERGY": "IREDA.NS",
    "INDIAN RENEWABLE ENERGY DEVELOPMENT AGENCY": "IREDA.NS",
    # New-age / technology listings
    "PROTEAN": "PROTEAN.NS",
    "PROTEAN EGOV": "PROTEAN.NS",
    "PROTEAN E GOV": "PROTEAN.NS",
    "PROTEAN E GOV TECH": "PROTEAN.NS",
    "PROTEAN EGOV TECHNOLOGIES": "PROTEAN.NS",
    "PROTEAN E GOV TECHNOLOGIES": "PROTEAN.NS",
    # Metals / industrials
    "TATASTEEL": "TATASTEEL.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "JSW STEEL": "JSWSTEEL.NS",
    "VEDL": "VEDL.NS",
    "VEDANTA": "VEDL.NS",
    "BEL": "BEL.NS",
    "BHARAT ELECTRONICS": "BEL.NS",
    "HAL": "HAL.NS",
    "HINDUSTAN AERONAUTICS": "HAL.NS",
    "IRCTC": "IRCTC.NS",
    "IRFC": "IRFC.NS",
    "DMART": "DMART.NS",
    "AVENUE SUPERMARTS": "DMART.NS",
}

EXCHANGE_SUFFIXES = (".NS", ".BO")

# Known listing renames/splits where the legacy ticker may no longer return quotes.
# These are only additional resolution attempts; no data is fabricated if all fail.
SYMBOL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "TATAMOTORS.NS": ("TMPV.NS", "TATAMOTORS.BO", "TMPV.BO"),
    "TATAMOTORS.BO": ("TMPV.BO", "TATAMOTORS.NS", "TMPV.NS"),
    "SBI.NS": ("SBIN.NS", "SBIN.BO", "SBI.BO"),
    "SBI.BO": ("SBIN.BO", "SBIN.NS", "SBI.NS"),
    "JAIPRAKASHPOWER.NS": ("JPPOWER.NS", "JPPOWER.BO"),
    "JAIPRAKASHPOWER.BO": ("JPPOWER.BO", "JPPOWER.NS"),
    "PROTEANEGOV.NS": ("PROTEAN.NS", "PROTEAN.BO"),
}


class SymbolResolutionError(ValueError):
    pass


def _normalize_input(symbol: str) -> str:
    """Normalize a user-supplied ticker/company name for alias matching.

    Handles punctuation (e.g. "Protean e-Gov Tech."), corporate suffixes and
    ampersands so extracted holdings from screenshots resolve consistently.
    """
    import re

    s = symbol.strip().upper().replace("&", " AND ")
    s = re.sub(r"[.,'’()\[\]{}:;/\-]", " ", s)
    s = re.sub(r"\b(LTD|LIMITED|PVT|PRIVATE|CO|COMPANY)\b", " ", s)
    return " ".join(s.split())


def _symbol_candidates(resolved: str) -> list[str]:
    """Return ordered ticker candidates: primary, alternate exchange, known fallbacks."""
    candidates = [resolved]
    upper = resolved.upper()
    if upper.endswith(".NS"):
        alt = resolved[:-3] + ".BO"
    elif upper.endswith(".BO"):
        alt = resolved[:-3] + ".NS"
    else:
        alt = None
    for candidate in (alt, *SYMBOL_FALLBACKS.get(upper, ())):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_symbol(symbol: str, exchange: str | None = None) -> tuple[str, str | None]:
    """Resolve user-provided symbol/name to a yfinance ticker.

    Returns (resolved_symbol, note). Raises SymbolResolutionError if ambiguous/unresolvable.
    Supports NSE/BSE suffixes and common unambiguous Indian names; defaults to NSE (.NS)
    when mapping is unambiguous and no exchange specified.
    """
    if not symbol or not symbol.strip():
        raise SymbolResolutionError("Empty symbol")

    raw = symbol.strip().upper()
    norm = _normalize_input(symbol)

    # Explicit suffix provided
    for suf in EXCHANGE_SUFFIXES:
        if raw.endswith(suf):
            return raw, None

    # Explicit exchange hint
    if exchange:
        ex = exchange.strip().upper()
        if ex in {"NSE", "NS"}:
            return f"{raw}.NS", "Resolved using explicit NSE exchange"
        if ex in {"BSE", "BO"}:
            return f"{raw}.BO", "Resolved using explicit BSE exchange"

    # Known unambiguous mapping
    if norm in COMMON_SYMBOL_MAP:
        return COMMON_SYMBOL_MAP[norm], "Resolved via known unambiguous symbol/name mapping (default NSE)"

    # Heuristic: a single plain alnum ticker-like token -> default NSE.
    # Multi-word unknown names are treated as ambiguous instead of guessing a ticker.
    compact = raw.replace(" ", "")
    if compact.isalnum() and 1 <= len(compact) <= 20 and len(norm.split()) == 1:
        return f"{compact}.NS", "Defaulted to NSE (.NS); verify if a different exchange was intended"

    raise SymbolResolutionError(
        f"Ambiguous or unresolvable symbol/name: '{symbol}'. Provide an explicit NSE/BSE ticker (e.g. SYMBOL.NS)."
    )


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except Exception:
        return None


def get_stock_price(symbol: str, exchange: str | None = None) -> MarketPrice:
    try:
        resolved, note = resolve_symbol(symbol, exchange)
    except SymbolResolutionError as exc:
        return MarketPrice(symbol=symbol, status="unresolved", message=str(exc))

    working_resolved = resolved
    last_error: Exception | None = None
    for candidate in _symbol_candidates(resolved):
        working_resolved = candidate
        try:
            ticker = yf.Ticker(candidate)
            # Prefer fast_info/latest history; do not claim live.
            price = None
            currency = None
            ts: datetime | None = None
            try:
                fi = getattr(ticker, "fast_info", None)
                if fi:
                    price = _safe_float(fi.get("lastPrice") or fi.get("last_price") or fi.get("regularMarketPrice"))
                    currency = fi.get("currency")
            except Exception:
                pass

            hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty:
                last_row = hist.iloc[-1]
                if price is None:
                    price = _safe_float(last_row.get("Close"))
                idx = hist.index[-1]
                if hasattr(idx, "to_pydatetime"):
                    ts = idx.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                try:
                    info = ticker.info or {}
                    currency = currency or info.get("currency")
                except Exception:
                    pass

            if price is None:
                # Try the next exchange/fallback ticker before giving up.
                continue

            used_note = note
            if candidate != resolved:
                used_note = ((note + " ") if note else "") + f"Resolved via fallback ticker {candidate}."

            return MarketPrice(
                symbol=symbol,
                resolved_symbol=candidate,
                latest_price=price,
                currency=currency,
                market_timestamp=ts,
                source="yfinance",
                status="ok",
                message=used_note,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("get_stock_price candidate %s failed for %s: %s", candidate, symbol, exc)
            continue

    if last_error is not None:
        return MarketPrice(
            symbol=symbol,
            resolved_symbol=working_resolved,
            source="yfinance",
            status="error",
            message=f"Failed to fetch price: {last_error}",
        )
    return MarketPrice(
        symbol=symbol,
        resolved_symbol=working_resolved,
        source="yfinance",
        status="missing_data",
        message=(note or "") + " Latest available price not found.",
    )


def get_historical_data(
    symbol: str, period: str = "1y", interval: str = "1d", exchange: str | None = None
) -> HistoricalData:
    try:
        resolved, note = resolve_symbol(symbol, exchange)
    except SymbolResolutionError as exc:
        return HistoricalData(symbol=symbol, period=period, interval=interval, status="unresolved", message=str(exc))

    working_resolved = resolved
    last_error: Exception | None = None
    for candidate in _symbol_candidates(resolved):
        working_resolved = candidate
        try:
            ticker = yf.Ticker(candidate)
            hist = ticker.history(period=period, interval=interval, auto_adjust=False)
            if hist is None or hist.empty:
                # Try the next exchange/fallback ticker before giving up.
                continue
            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else hist.columns[0]
            points: list[OHLCVPoint] = []
            for _, row in hist.iterrows():
                dt = row.get(date_col)
                if hasattr(dt, "date"):
                    date_str = dt.date().isoformat()
                else:
                    date_str = str(dt)[:10]
                vol = row.get("Volume")
                points.append(
                    OHLCVPoint(
                        date=date_str,
                        open=_safe_float(row.get("Open")),
                        high=_safe_float(row.get("High")),
                        low=_safe_float(row.get("Low")),
                        close=_safe_float(row.get("Close")),
                        volume=None if vol is None or (isinstance(vol, float) and pd.isna(vol)) else vol,
                    )
                )
            used_note = note
            if candidate != resolved:
                used_note = ((note + " ") if note else "") + f"Resolved via fallback ticker {candidate}."
            return HistoricalData(
                symbol=symbol,
                resolved_symbol=candidate,
                period=period,
                interval=interval,
                points=points,
                source="yfinance",
                status="ok",
                message=used_note,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("get_historical_data candidate %s failed for %s: %s", candidate, symbol, exc)
            continue

    if last_error is not None:
        return HistoricalData(
            symbol=symbol,
            resolved_symbol=working_resolved,
            period=period,
            interval=interval,
            status="error",
            message=f"Failed to fetch historical data: {last_error}",
        )
    return HistoricalData(
        symbol=symbol,
        resolved_symbol=working_resolved,
        period=period,
        interval=interval,
        status="missing_data",
        message=(note or "") + " No historical data returned.",
    )


def get_company_info(symbol: str, exchange: str | None = None) -> CompanyInfo:
    try:
        resolved, note = resolve_symbol(symbol, exchange)
    except SymbolResolutionError as exc:
        return CompanyInfo(symbol=symbol, status="unresolved", message=str(exc))

    working_resolved = resolved
    last_error: Exception | None = None
    for candidate in _symbol_candidates(resolved):
        working_resolved = candidate
        try:
            ticker = yf.Ticker(candidate)
            info: dict[str, Any] = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            fields = {
                "company_name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": _safe_float(info.get("marketCap")),
                "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
                "dividend_yield": _safe_float(info.get("dividendYield")),
            }
            missing = [k for k, v in fields.items() if v is None]
            status: Any = "ok" if any(v is not None for v in fields.values()) else "missing_data"
            if status == "missing_data":
                # Try the next exchange/fallback ticker before giving up.
                continue
            msg = note
            if status == "missing_data":
                msg = (note or "") + " Company information not available."
            elif candidate != resolved:
                msg = ((note + " ") if note else "") + f"Resolved via fallback ticker {candidate}."
            return CompanyInfo(
                symbol=symbol,
                resolved_symbol=candidate,
                company_name=fields["company_name"],
                sector=fields["sector"],
                industry=fields["industry"],
                market_cap=fields["market_cap"],
                pe_ratio=fields["pe_ratio"],
                dividend_yield=fields["dividend_yield"],
                source="yfinance",
                status=status,
                message=msg,
                missing_fields=missing,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("get_company_info candidate %s failed for %s: %s", candidate, symbol, exc)
            continue

    if last_error is not None:
        return CompanyInfo(
            symbol=symbol,
            resolved_symbol=working_resolved,
            source="yfinance",
            status="error",
            message=f"Failed to fetch company info: {last_error}",
        )
    return CompanyInfo(
        symbol=symbol,
        resolved_symbol=working_resolved,
        source="yfinance",
        status="missing_data",
        message=(note or "") + " Company information not available.",
        missing_fields=["company_name", "sector", "industry", "market_cap", "pe_ratio", "dividend_yield"],
    )
