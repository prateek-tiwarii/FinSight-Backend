import json
import uuid
from datetime import datetime, timezone

import pandas as pd
from langchain_core.messages import AIMessage

from app.api.routes import _derive_analysis_status
from app.models.analysis import (
    CompanyInfo,
    ConfirmedHolding,
    HistoricalData,
    MarketPrice,
    OHLCVPoint,
)
from app.services import market_data
from app.services.market_data import SymbolResolutionError, _symbol_candidates, resolve_symbol
from app.services.portfolio_analytics import analyze_portfolio


# ---------------------------------------------------------------------------
# Part 1: Indian symbol resolution
# ---------------------------------------------------------------------------

def test_resolve_common_indian_aliases():
    assert resolve_symbol("SBI")[0] == "SBIN.NS"
    assert resolve_symbol("JAIPRAKASHPOWER")[0] == "JPPOWER.NS"
    assert resolve_symbol("Indian Ren. Energy")[0] == "IREDA.NS"
    assert resolve_symbol("Protean e-Gov Tech.")[0] == "PROTEAN.NS"
    assert resolve_symbol("Tata Motors Ltd")[0] == "TATAMOTORS.NS"
    assert resolve_symbol("Reliance Industries Limited")[0] == "RELIANCE.NS"


def test_resolve_rejects_unresolvable_name():
    for value in ("??bad name??", "Acme Unknown Corp"):
        try:
            resolve_symbol(value)
        except SymbolResolutionError:
            continue
        raise AssertionError(f"Expected SymbolResolutionError for {value!r}")


def test_symbol_candidates_include_exchange_and_known_fallbacks():
    candidates = _symbol_candidates("SBI.NS")
    assert candidates[0] == "SBI.NS"
    assert "SBIN.NS" in candidates
    assert "SBI.BO" in candidates

    tata = _symbol_candidates("TATAMOTORS.NS")
    assert "TMPV.NS" in tata


def test_price_falls_back_to_alternate_valid_listing(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def fast_info(self):
            return {}

        @property
        def info(self):
            return {}

        def history(self, period="5d", interval="1d", auto_adjust=False):
            if self.symbol == "SBIN.NS":
                idx = pd.DatetimeIndex([pd.Timestamp("2025-01-02")])
                return pd.DataFrame({"Close": [550.0]}, index=idx)
            return pd.DataFrame()

    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)
    result = market_data.get_stock_price("SBI.NS")
    assert result.status == "ok"
    assert result.resolved_symbol == "SBIN.NS"
    assert result.latest_price == 550.0
    assert "fallback" in (result.message or "").lower()


# ---------------------------------------------------------------------------
# Part 3: aggregate completed/partial/failed status
# ---------------------------------------------------------------------------

def test_aggregate_status_reflects_failed_stage():
    metrics = {"summary": {"total_invested_amount": 1000.0}}
    pending_state = {
        "portfolio_metrics": metrics,
        "analysis_status": "completed",
        "research_status": "completed",
        "critic_status": "failed",
        "behavioral_status": "completed",
        "final_advisor_status": "completed",
    }
    assert _derive_analysis_status(pending_state, object(), object(), True) == "awaiting_user_input"
    assert _derive_analysis_status(pending_state, object(), object(), False) == "partial"

    completed_state = dict(pending_state, critic_status="completed")
    assert _derive_analysis_status(completed_state, object(), object(), False) == "completed"
    assert _derive_analysis_status({}, None, None, False) == "failed"


# ---------------------------------------------------------------------------
# Part 4: partial valuation must not mix unpriced cost basis into P&L
# ---------------------------------------------------------------------------

def _mock_partial_market(monkeypatch):
    def fake_price(symbol, exchange=None):
        if symbol == "AAA":
            return MarketPrice(
                symbol=symbol,
                resolved_symbol="AAA.NS",
                latest_price=110.0,
                currency="INR",
                market_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                status="ok",
                source="yfinance",
            )
        return MarketPrice(symbol=symbol, status="unresolved", message="No quote found")

    def fake_info(symbol, exchange=None):
        if symbol == "AAA":
            return CompanyInfo(
                symbol=symbol,
                resolved_symbol="AAA.NS",
                company_name="AAA Ltd",
                sector="Tech",
                industry="SW",
                status="ok",
                missing_fields=[],
            )
        return CompanyInfo(symbol=symbol, status="unresolved", message="No quote found")

    def fake_hist(symbol, period="1y", interval="1d", exchange=None):
        if symbol == "AAA":
            return HistoricalData(
                symbol=symbol,
                resolved_symbol="AAA.NS",
                period=period,
                interval=interval,
                status="ok",
                points=[
                    OHLCVPoint(date="2025-01-01", close=100.0),
                    OHLCVPoint(date="2025-01-02", close=105.0),
                    OHLCVPoint(date="2025-01-03", close=110.0),
                ],
            )
        return HistoricalData(symbol=symbol, period=period, interval=interval, status="unresolved", message="No history")

    monkeypatch.setattr("app.services.portfolio_analytics.get_stock_price", fake_price)
    monkeypatch.setattr("app.services.portfolio_analytics.get_company_info", fake_info)
    monkeypatch.setattr("app.services.portfolio_analytics.get_historical_data", fake_hist)


def test_partial_valuation_excludes_unpriced_cost_basis(monkeypatch):
    _mock_partial_market(monkeypatch)
    response = analyze_portfolio(
        holdings=[
            ConfirmedHolding(stock_symbol="AAA", quantity=10, average_purchase_price=100),
            ConfirmedHolding(stock_symbol="UNKNOWN CO", quantity=5, average_purchase_price=200),
        ]
    )
    summary = response.summary
    assert summary.total_invested_amount == 2000.0
    assert summary.analyzed_invested_amount == 1000.0
    assert summary.unvalued_invested_amount == 1000.0
    assert summary.total_current_value == 1100.0
    # P&L must compare priced value against priced cost basis only.
    assert summary.total_absolute_pnl == 100.0
    assert summary.overall_percentage_return == 10.0
    assert summary.is_partial is True
    assert summary.priced_holdings_count == 1
    assert summary.unresolved_holdings_count == 1
    assert summary.valuation_note


# ---------------------------------------------------------------------------
# Part 2: ToolNode must read/write the correct state message keys
# ---------------------------------------------------------------------------

ANALYST_JSON = {
    "portfolio_overview": "ok",
    "key_findings": [],
    "stock_level_insights": [],
    "sector_observations": [],
    "risk_observations": [],
    "supporting_data": [],
    "sources": [],
    "warnings": [],
}
RESEARCH_JSON = {
    "research_overview": "No relevant news found.",
    "company_research": [],
    "important_developments": [],
    "positive_signals": [],
    "negative_signals": [],
    "neutral_observations": ["No relevant news found."],
    "potential_concerns": [],
    "sources": [],
    "warnings": [],
}
CRITIC_JSON = {
    "critic_overview": "ok",
    "challenged_claims": [],
    "identified_risks": [],
    "contradictions": [],
    "missing_evidence": [],
    "severity_classification": {"overall": "low"},
    "recommendations_for_report": [],
    "supporting_sources": [],
    "warnings": [],
}
FINAL_JSON = {
    "report_title": "Report",
    "executive_summary": "Exec",
    "overall_assessment": "Overall",
    "portfolio_strengths": [],
    "portfolio_weaknesses": [],
    "performance_insights": [],
    "risk_assessment": {"summary": "r"},
    "research_summary": {"overview": "r"},
    "behavioral_assessment": {"overview": "b"},
    "preference_mismatches": [],
    "key_observations": [],
    "areas_requiring_attention": [],
    "critic_findings_summary": {"overview": "c"},
    "supporting_sources": [],
    "warnings": [],
    "generated_at": "2026-01-01T00:00:00Z",
}
QUESTIONNAIRE = {
    "investment_horizon": "3_5_years",
    "risk_tolerance": "moderate",
    "primary_investment_objective": "long_term_wealth_creation",
    "reaction_to_15_20pct_decline": "hold",
    "investment_experience": "intermediate",
    "additional_context": {},
}


class SequencedLLM:
    """Returns a fixed sequence of responses across repeated invocations."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class StaticLLM:
    def __init__(self, response):
        self.response = response

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.response


def _mock_market(monkeypatch):
    def fake_price(symbol, exchange=None):
        return MarketPrice(
            symbol=symbol,
            resolved_symbol=f"{symbol}.NS",
            latest_price=110.0,
            currency="INR",
            market_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            status="ok",
            source="yfinance",
        )

    def fake_hist(symbol, period="1y", interval="1d", exchange=None):
        return HistoricalData(
            symbol=symbol,
            resolved_symbol=f"{symbol}.NS",
            period=period,
            interval=interval,
            status="ok",
            points=[
                OHLCVPoint(date="2025-01-01", open=100, high=110, low=95, close=100, volume=1000),
                OHLCVPoint(date="2025-01-02", open=100, high=115, low=98, close=105, volume=1200),
                OHLCVPoint(date="2025-01-03", open=105, high=112, low=100, close=102, volume=1100),
            ],
        )

    def fake_info(symbol, exchange=None):
        return CompanyInfo(
            symbol=symbol,
            resolved_symbol=f"{symbol}.NS",
            company_name=f"{symbol} Ltd",
            sector="Tech",
            industry="SW",
            status="ok",
            missing_fields=[],
        )

    for module in ("app.services.portfolio_analytics", "app.services.market_data"):
        monkeypatch.setattr(f"{module}.get_stock_price", fake_price)
        monkeypatch.setattr(f"{module}.get_historical_data", fake_hist)
        monkeypatch.setattr(f"{module}.get_company_info", fake_info)


def test_toolnode_uses_agent_messages_key(monkeypatch):
    from app.graph.workflow import run_agent_analysis

    _mock_market(monkeypatch)
    analyst_llm = SequencedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_stock_price", "args": {"symbol": "AAA"}, "id": "call-1"}
                ],
            ),
            AIMessage(content=json.dumps(ANALYST_JSON)),
        ]
    )
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: analyst_llm)
    research_llm = SequencedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_company_news", "args": {"symbol": "AAA"}, "id": "call-2"}
                ],
            ),
            AIMessage(content=json.dumps(RESEARCH_JSON)),
        ]
    )
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: research_llm)

    def fake_news(symbol, company_name=None, lookback_days=30, max_articles=5, timeout_seconds=8):
        return {
            "query": "AAA",
            "symbol": symbol,
            "company_name": company_name,
            "articles": [
                {
                    "title": "AAA wins order",
                    "publisher": "Test Wire",
                    "url": "https://example.com/aaa-order",
                    "publication_date": "2026-01-01T00:00:00+00:00",
                    "snippet": "AAA announced a new order.",
                    "related_symbol": symbol,
                    "related_company": company_name,
                }
            ],
            "status": "ok",
            "message": None,
            "source": "google_news_rss",
            "warnings": [],
        }

    monkeypatch.setattr("app.services.news_service.search_company_news", fake_news)
    monkeypatch.setattr(
        "app.agents.risk_critic._build_llm",
        lambda: StaticLLM(AIMessage(content=json.dumps(CRITIC_JSON))),
    )
    monkeypatch.setattr(
        "app.agents.final_advisor._build_llm",
        lambda: StaticLLM(AIMessage(content=json.dumps(FINAL_JSON))),
    )

    out = run_agent_analysis(
        {
            "confirmed_holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}],
            "history_period": "1y",
            "history_interval": "1d",
            "behavioral_questionnaire": QUESTIONNAIRE,
        },
        thread_id=str(uuid.uuid4()),
    )

    assert out.get("analyst_findings") is not None
    assert out.get("critic_findings") is not None
    assert out.get("final_report") is not None
    summaries = out.get("tool_execution_summaries") or []
    assert any(
        s.get("tool_name") == "get_stock_price" and s.get("status") == "ok" for s in summaries
    ), f"Expected executed analyst tool summary, got: {summaries}"
    assert any(
        s.get("tool_name") == "search_company_news" and s.get("status") == "ok" for s in summaries
    ), f"Expected executed research tool summary, got: {summaries}"
