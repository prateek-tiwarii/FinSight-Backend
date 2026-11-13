import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.models.analysis import (
    ConfirmedHolding,
    HistoricalData,
    HoldingAnalysis,
    PortfolioAnalyzeResponse,
    PortfolioSummary,
    RiskMetrics,
)
from app.services.market_data import get_company_info, get_historical_data, get_stock_price

logger = logging.getLogger(__name__)


def calculate_invested_value(quantity: float, average_purchase_price: float) -> float:
    return float(quantity) * float(average_purchase_price)


def calculate_current_value(quantity: float, latest_price: float) -> float:
    return float(quantity) * float(latest_price)


def calculate_absolute_pnl(current_value: float, invested_value: float) -> float:
    return float(current_value) - float(invested_value)


def calculate_percentage_return(pnl: float, invested_value: float) -> float | None:
    if invested_value == 0:
        return None
    return (float(pnl) / float(invested_value)) * 100.0


def calculate_allocation_percent(value: float, total: float) -> float | None:
    if total == 0:
        return None
    return (float(value) / float(total)) * 100.0


def calculate_historical_volatility_annualized(
    close_prices: pd.Series, trading_days: int = 252
) -> float | None:
    """Annualized volatility = std(daily log/percent returns) * sqrt(trading_days).
    Uses daily percentage returns from close prices. Returns None if insufficient data."""
    if close_prices is None:
        return None
    s = pd.to_numeric(close_prices, errors="coerce").dropna()
    if len(s) < 3:
        return None
    returns = s.pct_change().dropna()
    if returns.empty:
        return None
    vol = float(returns.std(ddof=1) * math.sqrt(trading_days))
    if math.isnan(vol):
        return None
    return vol


def calculate_max_drawdown(close_prices: pd.Series) -> float | None:
    """Max drawdown as a negative fraction (e.g. -0.25 for 25% drawdown) or 0.
    Returns None if insufficient data."""
    if close_prices is None:
        return None
    s = pd.to_numeric(close_prices, errors="coerce").dropna()
    if len(s) < 2:
        return None
    running_max = s.cummax()
    drawdowns = (s / running_max) - 1.0
    mdd = float(drawdowns.min())
    if math.isnan(mdd):
        return None
    return mdd


def calculate_concentration(allocation_percents: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in allocation_percents if v is not None]
    if not vals:
        return None, None
    top = max(vals)
    # HHI on allocation percentages (0-100) -> 0-10000 scale
    hhi = float(sum((v ** 2) for v in vals))
    return float(top), hhi


def _historical_to_series(hist: HistoricalData) -> pd.Series | None:
    """Return close-price series indexed by date for date-aligned return calculations."""
    if hist.status != "ok" or not hist.points:
        return None
    data: dict[str, float] = {}
    for p in hist.points:
        if p.close is None or p.date is None:
            continue
        try:
            val = float(p.close)
        except Exception:
            continue
        # keep last value for duplicate dates
        data[str(p.date)[:10]] = val
    if len(data) < 2:
        return None
    s = pd.Series(data, dtype="float64")
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s


def analyze_portfolio(
    holdings: list[ConfirmedHolding], history_period: str = "1y", history_interval: str = "1d"
) -> PortfolioAnalyzeResponse:
    warnings: list[str] = []
    limitations: list[str] = [
        "Prices and company data are sourced from yfinance and may be delayed or reflect previous sessions; not represented as live market data.",
        "Historical metrics depend on available yfinance history; gaps/holidays/unavailable sessions may affect results.",
        "Sector information may be missing for some instruments; sector allocation includes only known sectors.",
    ]

    holding_results: list[HoldingAnalysis] = []
    per_holding_daily_returns: dict[str, pd.Series] = {}
    per_holding_weights: dict[str, float] = {}

    # First pass: fetch data and compute per-holding base metrics
    prelim: list[dict[str, Any]] = []
    for h in holdings:
        price_info = get_stock_price(h.stock_symbol, h.exchange)
        company = get_company_info(h.stock_symbol, h.exchange)
        hist = get_historical_data(h.stock_symbol, period=history_period, interval=history_interval, exchange=h.exchange)

        invested = calculate_invested_value(h.quantity, h.average_purchase_price)
        latest = price_info.latest_price if price_info.status == "ok" else None
        current = calculate_current_value(h.quantity, latest) if latest is not None else None
        pnl = calculate_absolute_pnl(current, invested) if current is not None else None
        pct = calculate_percentage_return(pnl, invested) if pnl is not None else None

        status: Any = "ok"
        message = None
        if price_info.status == "unresolved":
            status = "unresolved"
            message = price_info.message
        elif price_info.status in {"missing_data", "error"}:
            status = "error" if price_info.status == "error" else "missing_price"
            message = price_info.message
        elif latest is None:
            status = "missing_price"
            message = "Latest available price missing"

        if status != "ok":
            warnings.append(f"{h.stock_symbol}: {message or 'could not be fully analyzed'}")

        sector = company.sector if company and company.status in {"ok", "missing_data"} else None
        resolved = price_info.resolved_symbol or company.resolved_symbol or hist.resolved_symbol

        prelim.append(
            {
                "holding": h,
                "invested": invested,
                "latest": latest,
                "current": current,
                "pnl": pnl,
                "pct": pct,
                "status": status,
                "message": message,
                "company": company,
                "hist": hist,
                "sector": sector,
                "resolved": resolved,
            }
        )

    total_invested = float(sum(p["invested"] for p in prelim))
    # Totals/P&L must only compare like-for-like: value the holdings we actually
    # have a price for, otherwise including unvalued cost basis would understate P&L.
    priced = [p for p in prelim if p["current"] is not None]
    analyzed_invested = float(sum(p["invested"] for p in priced))
    total_current = float(sum(p["current"] for p in priced)) if priced else None
    total_pnl = (total_current - analyzed_invested) if total_current is not None else None
    overall_pct = calculate_percentage_return(total_pnl, analyzed_invested) if total_pnl is not None else None

    # Second pass: allocations and build results
    allocations: list[float] = []
    sector_totals: dict[str, float] = {}
    sector_base_total = 0.0

    for p in prelim:
        alloc = None
        if p["current"] is not None and total_current and total_current > 0:
            alloc = calculate_allocation_percent(p["current"], total_current)
            allocations.append(alloc)
        elif p["invested"] is not None and total_invested > 0 and total_current is None:
            # fallback for allocation display not meaningful without current; keep None
            alloc = None

        sector = p["sector"]
        if sector and p["current"] is not None and total_current and total_current > 0:
            sector_totals[sector] = sector_totals.get(sector, 0.0) + p["current"]
            sector_base_total += p["current"]
        elif sector and p["invested"] is not None and (not total_current):
            sector_totals[sector] = sector_totals.get(sector, 0.0) + p["invested"]
            sector_base_total += p["invested"]

        holding_results.append(
            HoldingAnalysis(
                input_symbol=p["holding"].stock_symbol,
                resolved_symbol=p["resolved"],
                quantity=p["holding"].quantity,
                average_purchase_price=p["holding"].average_purchase_price,
                invested_value=p["invested"],
                latest_price=p["latest"],
                current_value=p["current"],
                absolute_pnl=p["pnl"],
                percentage_return=p["pct"],
                allocation_percent=alloc,
                sector=sector,
                status=p["status"],
                message=p["message"],
                company=p["company"],
                historical_data=p["hist"],
            )
        )

        # collect for portfolio-level risk where usable
        series = _historical_to_series(p["hist"])
        if series is not None and p["current"] is not None and total_current and total_current > 0:
            key = p["resolved"] or p["holding"].stock_symbol
            per_holding_daily_returns[key] = series.pct_change().dropna()
            per_holding_weights[key] = p["current"] / total_current

    sector_allocation = {}
    if sector_base_total > 0:
        sector_allocation = {k: round((v / sector_base_total) * 100.0, 6) for k, v in sector_totals.items()}

    # Risk metrics
    top_conc, hhi = calculate_concentration(allocations)
    assumptions = [
        "Volatility calculated as standard deviation of daily percentage returns from close prices, annualized using sqrt(252 trading days).",
        "Portfolio volatility estimated using available holding histories aligned by date and weighted by current holding value proportions; holdings without sufficient history are excluded from this estimate.",
        "Max drawdown computed on the weighted portfolio value series derived from aligned historical close prices (where sufficient data exists).",
        "Concentration based on current-value allocation percentages of analyzable holdings only.",
    ]
    vol_note = None
    portfolio_vol = None
    mdd = None
    mdd_note = None

    try:
        if len(per_holding_daily_returns) >= 1 and per_holding_weights:
            # align returns on common dates
            ret_df = pd.DataFrame(per_holding_daily_returns).dropna(how="any")
            if len(ret_df) >= 3:
                weights = np.array([per_holding_weights.get(c, 0.0) for c in ret_df.columns], dtype="float64")
                wsum = weights.sum()
                if wsum > 0:
                    weights = weights / wsum
                    port_ret = ret_df.values @ weights
                    port_ret_s = pd.Series(port_ret)
                    portfolio_vol = float(port_ret_s.std(ddof=1) * math.sqrt(252))
                    if math.isnan(portfolio_vol):
                        portfolio_vol = None
                    # portfolio value series proxy (start at 1)
                    port_value = (1.0 + port_ret_s).cumprod()
                    mdd = calculate_max_drawdown(port_value)
            else:
                vol_note = "Insufficient overlapping historical data to compute portfolio volatility."
                mdd_note = "Insufficient overlapping historical data to compute max drawdown."
        else:
            vol_note = "Insufficient historical data to compute portfolio volatility."
            mdd_note = "Insufficient historical data to compute max drawdown."
    except Exception as exc:
        logger.exception("Risk metric calculation failed: %s", exc)
        vol_note = "Portfolio volatility could not be computed due to a calculation/data issue."
        mdd_note = "Max drawdown could not be computed due to a calculation/data issue."

    if portfolio_vol is None and vol_note is None:
        vol_note = "Portfolio volatility unavailable (insufficient or unaligned history)."
    if mdd is None and mdd_note is None:
        mdd_note = "Max drawdown unavailable (insufficient or unaligned history)."

    analyzed_count = sum(1 for r in holding_results if r.status == "ok")
    failed_count = len(holding_results) - analyzed_count
    priced_count = len(priced)
    missing_price_count = sum(1 for r in holding_results if r.status == "missing_price")
    unresolved_count = sum(1 for r in holding_results if r.status == "unresolved")
    failed_status_count = sum(1 for r in holding_results if r.status == "error")
    unvalued_invested = total_invested - analyzed_invested
    is_partial = priced_count < len(holding_results)
    valuation_note = None
    if failed_count:
        warnings.append(f"{failed_count} holding(s) could not be fully analyzed; see holding statuses/messages.")
    if is_partial:
        valuation_note = (
            f"Partial valuation: {priced_count} of {len(holding_results)} holdings have a usable price. "
            "Totals, P&L and returns below include only priced holdings; "
            f"invested cost for holdings without a price is excluded."
        )
        warnings.append(valuation_note)

    summary = PortfolioSummary(
        total_invested_amount=round(total_invested, 6),
        total_current_value=round(total_current, 6) if total_current is not None else None,
        total_absolute_pnl=round(total_pnl, 6) if total_pnl is not None else None,
        overall_percentage_return=round(overall_pct, 6) if overall_pct is not None else None,
        analyzed_holdings_count=analyzed_count,
        unresolved_or_failed_holdings_count=failed_count,
        sector_allocation=sector_allocation,
        total_holdings_count=len(holding_results),
        priced_holdings_count=priced_count,
        missing_price_holdings_count=missing_price_count,
        unresolved_holdings_count=unresolved_count,
        failed_holdings_count=failed_status_count,
        analyzed_invested_amount=round(analyzed_invested, 6) if priced_count else None,
        unvalued_invested_amount=round(unvalued_invested, 6) if is_partial else 0.0,
        is_partial=is_partial,
        valuation_note=valuation_note,
    )

    risk = RiskMetrics(
        portfolio_volatility_annualized=round(portfolio_vol, 8) if portfolio_vol is not None else None,
        portfolio_volatility_note=vol_note,
        max_drawdown=round(mdd, 8) if mdd is not None else None,
        max_drawdown_note=mdd_note,
        concentration_top_holding_percent=round(top_conc, 6) if top_conc is not None else None,
        concentration_hhi=round(hhi, 6) if hhi is not None else None,
        assumptions=assumptions,
    )

    return PortfolioAnalyzeResponse(
        holdings=holding_results,
        summary=summary,
        risk=risk,
        data_source="yfinance",
        generated_at=datetime.now(timezone.utc),
        warnings=warnings,
        limitations=limitations,
    )
