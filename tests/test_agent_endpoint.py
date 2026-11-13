import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from app.models.analysis import CompanyInfo, HistoricalData, MarketPrice, OHLCVPoint
from tests.helpers import make_guest_client

client = make_guest_client()


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


def _patch_market_data(monkeypatch):
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
    # tools also call market_data module; patch them to be safe if ever used
    monkeypatch.setattr("app.services.market_data.get_stock_price", fake_price)
    monkeypatch.setattr("app.services.market_data.get_historical_data", fake_hist)
    monkeypatch.setattr("app.services.market_data.get_company_info", fake_info)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    def with_structured_output(self, schema):
        return self


QUESTIONNAIRE = {
    "investment_horizon": "3_5_years",
    "risk_tolerance": "moderate",
    "primary_investment_objective": "long_term_wealth_creation",
    "reaction_to_15_20pct_decline": "hold",
    "investment_experience": "intermediate",
    "additional_context": {},
}


def test_agent_analyze_endpoint_success(monkeypatch):
    _patch_market_data(monkeypatch)
    final_json = {
        "portfolio_overview": "Portfolio is up with concentration in tested names.",
        "key_findings": ["Total P&L positive"],
        "stock_level_insights": [{"symbol": "AAA.NS", "observation": "Gain", "interpretation": "Outperformed cost basis"}],
        "sector_observations": ["Mixed sectors"],
        "risk_observations": ["Moderate concentration"],
        "supporting_data": [{"description": "overall_return", "value": 10.0, "source": "calculation"}],
        "sources": [{"name": "yfinance", "type": "market_data", "detail": "mocked"}],
        "warnings": [],
    }
    research_json = {
        "research_overview": "No major recent developments found in retrieved results.",
        "company_research": [],
        "important_developments": [],
        "positive_signals": [],
        "negative_signals": [],
        "neutral_observations": ["No relevant news found."],
        "potential_concerns": [],
        "sources": [],
        "warnings": ["No relevant news found for holdings in lookback period."],
    }
    critic_json = {
        "critic_overview": "Review complete.",
        "challenged_claims": [],
        "identified_risks": [{"title": "Concentration", "description": "Top holding concentrated", "severity": "medium", "evidence": [], "related_stock_or_sector": "AAA.NS", "reasoning_summary": "Concentration increases idiosyncratic risk."}],
        "contradictions": [],
        "missing_evidence": [],
        "severity_classification": {"overall": "medium"},
        "recommendations_for_report": ["Highlight concentration."],
        "supporting_sources": [],
        "warnings": [],
    }
    final_report_json = {
        "report_title": "FinSight AI Portfolio Intelligence Report",
        "executive_summary": "Portfolio shows gains with moderate concentration.",
        "overall_assessment": "Generally positive performance with noted concentration risk.",
        "portfolio_strengths": ["Positive overall return"],
        "portfolio_weaknesses": ["Concentration risk"],
        "performance_insights": [{"area": "returns", "insight": "Positive P&L", "evidence": "deterministic metrics"}],
        "risk_assessment": {"summary": "Moderate risk", "key_risks": ["Concentration"]},
        "research_summary": {"overview": "No major recent developments found.", "sources": []},
        "behavioral_assessment": {"overview": "Preferences compared to portfolio."},
        "preference_mismatches": [],
        "key_observations": ["Concentration noted."],
        "areas_requiring_attention": ["Monitor concentration."],
        "critic_findings_summary": {"overview": "Concentration flagged."},
        "supporting_sources": [],
        "warnings": [],
        "generated_at": "2026-01-01T00:00:00Z",
    }
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(final_json))]))
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(research_json))]))
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(critic_json))]))
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(final_report_json))]))

    payload = {
        "holdings": [
            {"stock_symbol": "AAA", "quantity": 10, "average_purchase_price": 100},
            {"stock_symbol": "BBB", "quantity": 20, "average_purchase_price": 50},
        ],
        "history_period": "1y",
        "history_interval": "1d",
        "max_tool_iterations": 2,
        "questionnaire_responses": QUESTIONNAIRE,
    }
    resp = client.post("/api/agent/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_status"] == "completed"
    assert data["portfolio_metrics"]["summary"]["total_current_value"] == 2200.0
    assert data["analyst_findings"]["portfolio_overview"].startswith("Portfolio")
    assert isinstance(data["tool_execution_summaries"], list)
    assert len(data["sources"]) >= 1
    # M4/M5 fields present/backward-compatible extension
    assert "research_status" in data
    assert "research_findings" in data
    assert data["critic_findings"] is not None
    assert data["final_report"] is not None
    assert data["pending_questionnaire"] is False
    assert data["thread_id"] is not None


def test_agent_analyze_returns_metrics_when_agent_fails(monkeypatch):
    _patch_market_data(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr("app.graph.nodes._build_llm", boom)
    # downstream agents should also fail gracefully if reached
    monkeypatch.setattr("app.agents.research_agent._build_llm", boom)
    monkeypatch.setattr("app.agents.risk_critic._build_llm", boom)
    monkeypatch.setattr("app.agents.final_advisor._build_llm", boom)

    payload = {
        "holdings": [{"stock_symbol": "AAA", "quantity": 10, "average_purchase_price": 100}],
        "history_period": "1y",
        "history_interval": "1d",
        "questionnaire_responses": QUESTIONNAIRE,
    }
    resp = client.post("/api/agent/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # deterministic metrics should still be present even if agents fail
    assert data["portfolio_metrics"] is not None
    assert data["analysis_status"] in {"partial", "failed", "completed"}
    assert len(data["warnings"]) >= 1
    assert data["pending_questionnaire"] is False
    assert data.get("final_report") is not None  # fallback/partial report may be produced
