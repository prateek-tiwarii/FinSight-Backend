import math

import numpy as np
import pandas as pd

from app.models.analysis import CompanyInfo, ConfirmedHolding, HistoricalData, MarketPrice, OHLCVPoint
from app.services.portfolio_analytics import (
    analyze_portfolio,
    calculate_absolute_pnl,
    calculate_allocation_percent,
    calculate_concentration,
    calculate_current_value,
    calculate_historical_volatility_annualized,
    calculate_invested_value,
    calculate_max_drawdown,
    calculate_percentage_return,
)


def test_invested_and_current_value():
    assert calculate_invested_value(10, 250.5) == 2505.0
    assert calculate_current_value(10, 300) == 3000.0


def test_pnl_and_percentage_return():
    pnl = calculate_absolute_pnl(3000, 2500)
    assert pnl == 500
    assert calculate_percentage_return(pnl, 2500) == 20.0
    assert calculate_percentage_return(10, 0) is None


def test_allocation_percent():
    assert calculate_allocation_percent(2500, 10000) == 25.0
    assert calculate_allocation_percent(1, 0) is None


def test_concentration_metrics():
    top, hhi = calculate_concentration([60.0, 25.0, 15.0])
    assert top == 60.0
    assert hhi == (60.0**2 + 25.0**2 + 15.0**2)
    assert calculate_concentration([]) == (None, None)


def test_historical_volatility_annualized():
    # Construct a price series with known daily returns variability
    prices = pd.Series([100, 101, 99, 102, 100, 103, 101, 104])
    vol = calculate_historical_volatility_annualized(prices)
    assert vol is not None and vol > 0
    assert calculate_historical_volatility_annualized(pd.Series([100, 101])) is None


def test_max_drawdown():
    prices = pd.Series([100, 120, 90, 95, 80, 110])
    mdd = calculate_max_drawdown(prices)
    # peak 120 to trough 80 => -1/3
    assert mdd is not None
    assert math.isclose(mdd, (80 / 120) - 1.0, rel_tol=1e-9)
    assert calculate_max_drawdown(pd.Series([100])) is None


def _mk_price(symbol: str, resolved: str, latest: float, status: str = "ok", msg: str | None = None):
    return MarketPrice(
        symbol=symbol,
        resolved_symbol=resolved if status != "unresolved" else None,
        latest_price=latest if status == "ok" else None,
        currency="INR" if status == "ok" else None,
        source="yfinance",
        status=status,
        message=msg,
    )


def _mk_info(symbol: str, resolved: str, sector: str = "Tech"):
    return CompanyInfo(symbol=symbol, resolved_symbol=resolved, sector=sector, status="ok", missing_fields=[])


def _mk_hist(symbol: str, resolved: str, rows, status: str = "ok", msg: str | None = None):
    # rows: list of (date, close) or None for gaps
    points = []
    for d, c in rows:
        points.append(OHLCVPoint(date=d, open=c, high=c, low=c, close=c, volume=1))
    return HistoricalData(
        symbol=symbol,
        resolved_symbol=resolved if status != "unresolved" else None,
        period="1y",
        interval="1d",
        points=points if status == "ok" else [],
        status=status,
        message=msg,
    )


def _patch_market(monkeypatch, price_map, hist_map, info_map):
    monkeypatch.setattr("app.services.portfolio_analytics.get_stock_price", lambda s, e=None: price_map[s])
    monkeypatch.setattr("app.services.portfolio_analytics.get_historical_data", lambda s, period="1y", interval="1d", exchange=None: hist_map[s])
    monkeypatch.setattr("app.services.portfolio_analytics.get_company_info", lambda s, e=None: info_map[s])


def _expected_port_risk(closes_by_symbol_dates, weights_by_symbol):
    """Compute expected vol/mdd from date-aligned intersection with renormalized weights."""
    ret_dfs = {}
    for sym, date_close in closes_by_symbol_dates.items():
        s = pd.Series({pd.to_datetime(d): c for d, c in date_close}).sort_index()
        ret_dfs[sym] = s.pct_change().dropna()
    df = pd.DataFrame(ret_dfs).dropna(how="any")
    if len(df) < 3:
        return None, None, len(df)
    w = np.array([weights_by_symbol[c] for c in df.columns], dtype="float64")
    w = w / w.sum()
    port_ret = pd.Series(df.values @ w)
    vol = float(port_ret.std(ddof=1) * math.sqrt(252))
    mdd = calculate_max_drawdown((1.0 + port_ret).cumprod())
    return vol, mdd, len(df)


def test_risk_different_trading_calendars_uses_date_intersection_and_renormalizes(monkeypatch):
    # AAA trades on extra date; BBB missing 2025-01-03 -> intersection excludes that date
    aaa_rows = [("2025-01-01", 100), ("2025-01-02", 102), ("2025-01-03", 101), ("2025-01-06", 103), ("2025-01-07", 104)]
    bbb_rows = [("2025-01-01", 50), ("2025-01-02", 51), ("2025-01-06", 52), ("2025-01-07", 53), ("2025-01-08", 54)]
    price_map = {"AAA": _mk_price("AAA", "AAA.NS", 104), "BBB": _mk_price("BBB", "BBB.NS", 53)}
    hist_map = {"AAA": _mk_hist("AAA", "AAA.NS", aaa_rows), "BBB": _mk_hist("BBB", "BBB.NS", bbb_rows)}
    info_map = {"AAA": _mk_info("AAA", "AAA.NS"), "BBB": _mk_info("BBB", "BBB.NS", "Fin")}
    _patch_market(monkeypatch, price_map, hist_map, info_map)

    holdings = [ConfirmedHolding(stock_symbol="AAA", quantity=10, average_purchase_price=100),
                ConfirmedHolding(stock_symbol="BBB", quantity=10, average_purchase_price=50)]
    res = analyze_portfolio(holdings)

    assert len(res.holdings) == 2 and all(h.status == "ok" for h in res.holdings)
    # weights by current value 1040 vs 530
    w = {"AAA.NS": 1040 / 1570, "BBB.NS": 530 / 1570}
    exp_vol, exp_mdd, n = _expected_port_risk({"AAA.NS": aaa_rows, "BBB.NS": bbb_rows}, w)
    assert n >= 3
    assert res.risk.portfolio_volatility_annualized is not None
    assert res.risk.max_drawdown is not None
    assert math.isclose(res.risk.portfolio_volatility_annualized, exp_vol, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(res.risk.max_drawdown, exp_mdd, rel_tol=1e-6, abs_tol=1e-8)
    # sanity: must not equal naive positional alignment (which would differ due to BBB shift)
    s_aaa = pd.Series([100,102,101,103,104], dtype="float64").pct_change().dropna()
    s_bbb = pd.Series([50,51,52,53,54], dtype="float64").pct_change().dropna()  # positional, wrong
    naive_df = pd.DataFrame({"AAA.NS": s_aaa, "BBB.NS": s_bbb}).dropna(how="any")
    naive_w = np.array([w["AAA.NS"], w["BBB.NS"]]); naive_w /= naive_w.sum()
    naive_vol = float(pd.Series(naive_df.values @ naive_w).std(ddof=1) * math.sqrt(252))
    assert not math.isclose(res.risk.portfolio_volatility_annualized, naive_vol, rel_tol=1e-6)


def test_risk_partial_history_gaps_insufficient_common_dates(monkeypatch):
    # after date-intersection/dropna only 2 common return points -> insufficient
    aaa_rows = [("2025-01-01", 100), ("2025-01-02", 101), ("2025-01-03", 102), ("2025-01-06", 103)]
    bbb_rows = [("2025-01-01", 50), ("2025-01-02", 51), ("2025-01-06", 52), ("2025-01-07", 53)]  # missing 01-03
    price_map = {"AAA": _mk_price("AAA", "AAA.NS", 103), "BBB": _mk_price("BBB", "BBB.NS", 53)}
    hist_map = {"AAA": _mk_hist("AAA", "AAA.NS", aaa_rows), "BBB": _mk_hist("BBB", "BBB.NS", bbb_rows)}
    info_map = {"AAA": _mk_info("AAA", "AAA.NS"), "BBB": _mk_info("BBB", "BBB.NS")}
    _patch_market(monkeypatch, price_map, hist_map, info_map)
    res = analyze_portfolio([ConfirmedHolding(stock_symbol="AAA", quantity=1, average_purchase_price=100),
                             ConfirmedHolding(stock_symbol="BBB", quantity=1, average_purchase_price=50)])
    assert res.risk.portfolio_volatility_annualized is None
    assert res.risk.max_drawdown is None
    assert "Insufficient overlapping historical data" in (res.risk.portfolio_volatility_note or "")
    assert "Insufficient overlapping historical data" in (res.risk.max_drawdown_note or "")


def test_risk_one_holding_unavailable_excluded_but_present_single_usable_insufficient(monkeypatch):
    # BBB history unavailable -> must remain in response, excluded from risk; AAA alone has <3 returns
    aaa_rows = [("2025-01-01", 100), ("2025-01-02", 101), ("2025-01-03", 102)]  # 2 returns
    price_map = {"AAA": _mk_price("AAA", "AAA.NS", 102), "BBB": _mk_price("BBB", "BBB.NS", 55)}
    hist_map = {
        "AAA": _mk_hist("AAA", "AAA.NS", aaa_rows),
        "BBB": _mk_hist("BBB", "BBB.NS", [], status="missing_data", msg="No historical data"),
    }
    info_map = {"AAA": _mk_info("AAA", "AAA.NS"), "BBB": _mk_info("BBB", "BBB.NS")}
    _patch_market(monkeypatch, price_map, hist_map, info_map)
    res = analyze_portfolio([ConfirmedHolding(stock_symbol="AAA", quantity=10, average_purchase_price=100),
                             ConfirmedHolding(stock_symbol="BBB", quantity=10, average_purchase_price=50)])
    assert len(res.holdings) == 2
    bbb = next(h for h in res.holdings if h.input_symbol == "BBB")
    assert bbb.historical_data.status == "missing_data"
    assert bbb.status == "ok"  # price ok, only history missing
    # risk should not be computed from AAA alone (needs >=3 aligned return rows)
    assert res.risk.portfolio_volatility_annualized is None
    assert res.risk.max_drawdown is None
    assert res.risk.portfolio_volatility_note is not None
    # AAA had current value so it was candidate but insufficient returns; BBB excluded (no series)


def test_risk_single_usable_holding_sufficient_history(monkeypatch):
    # single holding with enough history -> vol/mdd computed from its returns (renormalized weight 1)
    rows = [("2025-01-01", 100), ("2025-01-02", 102), ("2025-01-03", 101), ("2025-01-06", 103), ("2025-01-07", 99)]
    price_map = {"AAA": _mk_price("AAA", "AAA.NS", 99)}
    hist_map = {"AAA": _mk_hist("AAA", "AAA.NS", rows)}
    info_map = {"AAA": _mk_info("AAA", "AAA.NS")}
    _patch_market(monkeypatch, price_map, hist_map, info_map)
    res = analyze_portfolio([ConfirmedHolding(stock_symbol="AAA", quantity=1, average_purchase_price=100)])
    exp_vol, exp_mdd, n = _expected_port_risk({"AAA.NS": rows}, {"AAA.NS": 1.0})
    assert n >= 3
    assert res.risk.portfolio_volatility_annualized is not None
    assert math.isclose(res.risk.portfolio_volatility_annualized, exp_vol, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(res.risk.max_drawdown, exp_mdd, rel_tol=1e-6, abs_tol=1e-8)


def test_risk_all_holdings_unavailable(monkeypatch):
    price_map = {
        "AAA": _mk_price("AAA", "AAA.NS", 0, status="missing_data", msg="no price"),
        "BBB": _mk_price("BBB", "BBB.NS", 0, status="unresolved", msg="unresolvable"),
    }
    hist_map = {
        "AAA": _mk_hist("AAA", "AAA.NS", [], status="missing_data", msg="no hist"),
        "BBB": _mk_hist("BBB", "BBB.NS", [], status="unresolved", msg="unresolvable"),
    }
    info_map = {
        "AAA": _mk_info("AAA", "AAA.NS"),
        "BBB": CompanyInfo(symbol="BBB", status="unresolved", message="unresolvable"),
    }
    _patch_market(monkeypatch, price_map, hist_map, info_map)
    res = analyze_portfolio([ConfirmedHolding(stock_symbol="AAA", quantity=1, average_purchase_price=100),
                             ConfirmedHolding(stock_symbol="BBB", quantity=1, average_purchase_price=50)])
    assert len(res.holdings) == 2  # not silently removed
    assert res.summary.total_current_value is None
    assert res.risk.portfolio_volatility_annualized is None
    assert res.risk.max_drawdown is None
    assert res.risk.portfolio_volatility_note is not None
    assert res.risk.max_drawdown_note is not None
    assert res.summary.unresolved_or_failed_holdings_count == 2
    assert len(res.warnings) >= 1


def test_risk_no_silent_zero_returns_on_missing(monkeypatch):
    # Missing dates must be excluded via intersection, not filled as 0% returns.
    # Common dates: 01-01,01-03,01-06,01-07 (BBB missing 01-02) -> 4 prices => 3 returns
    aaa_rows = [("2025-01-01", 100), ("2025-01-02", 110), ("2025-01-03", 120), ("2025-01-06", 130), ("2025-01-07", 125)]
    bbb_rows = [("2025-01-01", 50), ("2025-01-03", 55), ("2025-01-06", 60), ("2025-01-07", 58)]  # missing 01-02
    price_map = {"AAA": _mk_price("AAA", "AAA.NS", 125), "BBB": _mk_price("BBB", "BBB.NS", 58)}
    hist_map = {"AAA": _mk_hist("AAA", "AAA.NS", aaa_rows), "BBB": _mk_hist("BBB", "BBB.NS", bbb_rows)}
    info_map = {"AAA": _mk_info("AAA", "AAA.NS"), "BBB": _mk_info("BBB", "BBB.NS")}
    _patch_market(monkeypatch, price_map, hist_map, info_map)
    res = analyze_portfolio([ConfirmedHolding(stock_symbol="AAA", quantity=1, average_purchase_price=100),
                             ConfirmedHolding(stock_symbol="BBB", quantity=1, average_purchase_price=50)])
    # If 01-02 were treated as 0 return for BBB, result would differ; verify equals date-intersection result
    w = {"AAA.NS": 125 / 183, "BBB.NS": 58 / 183}
    exp_vol, exp_mdd, n = _expected_port_risk({"AAA.NS": aaa_rows, "BBB.NS": bbb_rows}, w)
    assert n >= 3
    assert math.isclose(res.risk.portfolio_volatility_annualized, exp_vol, rel_tol=1e-6, abs_tol=1e-8)
    # demonstrate zero-fill would be different
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    a = pd.Series([100, 110, 120, 130, 125], index=idx).pct_change()
    b = pd.Series([50, None, 55, 60, 58], index=idx, dtype="float64").pct_change().fillna(0.0)
    df = pd.DataFrame({"AAA.NS": a, "BBB.NS": b}).dropna()
    wv = np.array([w["AAA.NS"], w["BBB.NS"]]); wv /= wv.sum()
    zero_fill_vol = float(pd.Series(df.values @ wv).std(ddof=1) * math.sqrt(252))
    assert not math.isclose(res.risk.portfolio_volatility_annualized, zero_fill_vol, rel_tol=1e-6)

