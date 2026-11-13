from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models.analysis import CompanyInfo, HistoricalData, MarketPrice, OHLCVPoint
from tests.helpers import prime_csrf

client = TestClient(app)
prime_csrf(client)


def _mock_hist(symbol, resolved, period="1y", interval="1d"):
    return HistoricalData(
        symbol=symbol,
        resolved_symbol=resolved,
        period=period,
        interval=interval,
        status="ok",
        points=[
            OHLCVPoint(date="2025-01-01", open=100, high=110, low=95, close=100, volume=1000),
            OHLCVPoint(date="2025-01-02", open=100, high=115, low=98, close=105, volume=1200),
            OHLCVPoint(date="2025-01-03", open=105, high=112, low=100, close=102, volume=1100),
            OHLCVPoint(date="2025-01-06", open=102, high=118, low=101, close=110, volume=1300),
            OHLCVPoint(date="2025-01-07", open=110, high=111, low=104, close=108, volume=900),
        ],
    )


def test_analyze_portfolio_success_with_mocks(monkeypatch):
    def fake_price(symbol, exchange=None):
        resolved = f"{symbol}.NS"
        price = 110.0 if symbol == "AAA" else 55.0
        return MarketPrice(
            symbol=symbol,
            resolved_symbol=resolved,
            latest_price=price,
            currency="INR",
            market_timestamp=datetime(2026, 1, 10, tzinfo=timezone.utc),
            status="ok",
            source="yfinance",
        )

    def fake_hist(symbol, period="1y", interval="1d", exchange=None):
        return _mock_hist(symbol, f"{symbol}.NS", period, interval)

    def fake_info(symbol, exchange=None):
        sector = "Technology" if symbol == "AAA" else "Finance"
        return CompanyInfo(
            symbol=symbol,
            resolved_symbol=f"{symbol}.NS",
            company_name=f"{symbol} Ltd",
            sector=sector,
            industry="Test",
            market_cap=1000.0,
            pe_ratio=20.0,
            dividend_yield=0.01,
            status="ok",
            missing_fields=[],
        )

    monkeypatch.setattr("app.services.portfolio_analytics.get_stock_price", fake_price)
    monkeypatch.setattr("app.services.portfolio_analytics.get_historical_data", fake_hist)
    monkeypatch.setattr("app.services.portfolio_analytics.get_company_info", fake_info)

    payload = {
        "holdings": [
            {"stock_symbol": "AAA", "quantity": 10, "average_purchase_price": 100},
            {"stock_symbol": "BBB", "quantity": 20, "average_purchase_price": 50},
        ],
        "history_period": "1y",
        "history_interval": "1d",
    }
    resp = client.post("/api/portfolio/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["data_source"] == "yfinance"
    assert data["summary"]["total_invested_amount"] == 2000.0
    assert data["summary"]["total_current_value"] == 2200.0
    assert data["summary"]["total_absolute_pnl"] == 200.0
    assert data["summary"]["overall_percentage_return"] == 10.0
    assert data["summary"]["analyzed_holdings_count"] == 2
    assert len(data["holdings"]) == 2
    assert data["holdings"][0]["invested_value"] == 1000.0
    assert data["holdings"][0]["current_value"] == 1100.0
    assert data["holdings"][0]["historical_data"]["status"] == "ok"
    assert "concentration_note" in data["risk"]


def test_analyze_portfolio_handles_unresolved_symbol(monkeypatch):
    from app.models.analysis import MarketPrice

    def fake_price(symbol, exchange=None):
        return MarketPrice(symbol=symbol, status="unresolved", message="unresolvable")

    def fake_hist(symbol, period="1y", interval="1d", exchange=None):
        return HistoricalData(symbol=symbol, period=period, interval=interval, status="unresolved", message="unresolvable")

    def fake_info(symbol, exchange=None):
        return CompanyInfo(symbol=symbol, status="unresolved", message="unresolvable")

    monkeypatch.setattr("app.services.portfolio_analytics.get_stock_price", fake_price)
    monkeypatch.setattr("app.services.portfolio_analytics.get_historical_data", fake_hist)
    monkeypatch.setattr("app.services.portfolio_analytics.get_company_info", fake_info)

    payload = {"holdings": [{"stock_symbol": "??bad name??", "quantity": 1, "average_purchase_price": 10}]}
    resp = client.post("/api/portfolio/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["holdings"][0]["status"] == "unresolved"
    assert data["summary"]["unresolved_or_failed_holdings_count"] == 1
    assert data["summary"]["total_current_value"] is None
