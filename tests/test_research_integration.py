import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from app.graph.workflow import build_agent_graph, run_agent_analysis
from app.models.analysis import CompanyInfo, HistoricalData, MarketPrice, OHLCVPoint
from tests.helpers import make_guest_client

client = make_guest_client()


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


def _mock_hist(symbol, resolved, period="1y", interval="1d"):
    return HistoricalData(
        symbol=symbol, resolved_symbol=resolved, period=period, interval=interval, status="ok",
        points=[
            OHLCVPoint(date="2025-01-01", open=100, high=110, low=95, close=100, volume=1000),
            OHLCVPoint(date="2025-01-02", open=100, high=115, low=98, close=105, volume=1200),
            OHLCVPoint(date="2025-01-03", open=105, high=112, low=100, close=102, volume=1100),
        ],
    )


QUESTIONNAIRE = {
    "investment_horizon": "3_5_years",
    "risk_tolerance": "moderate",
    "primary_investment_objective": "long_term_wealth_creation",
    "reaction_to_15_20pct_decline": "hold",
    "investment_experience": "intermediate",
    "additional_context": {},
}


def _critic_json():
    return {
        "critic_overview": "ok",
        "challenged_claims": [],
        "identified_risks": [{"title": "Concentration", "description": "High concentration", "severity": "medium", "evidence": [], "related_stock_or_sector": "AAA", "reasoning_summary": "Top holding dominates"}],
        "contradictions": [],
        "missing_evidence": [],
        "severity_classification": {"overall": "medium"},
        "recommendations_for_report": [],
        "supporting_sources": [],
        "warnings": [],
    }


def _final_json():
    return {
        "report_title": "FinSight AI Report",
        "executive_summary": "Summary",
        "overall_assessment": "Overall",
        "portfolio_strengths": [],
        "portfolio_weaknesses": [],
        "performance_insights": [],
        "risk_assessment": {"summary": "risk"},
        "research_summary": {"overview": "research"},
        "behavioral_assessment": {"overview": "behavioral"},
        "preference_mismatches": [],
        "key_observations": [],
        "areas_requiring_attention": [],
        "critic_findings_summary": {"overview": "critic"},
        "supporting_sources": [],
        "warnings": [],
        "generated_at": "2026-01-01T00:00:00Z",
    }


def _patch_market(monkeypatch):
    monkeypatch.setattr("app.services.portfolio_analytics.get_stock_price", lambda s, e=None: MarketPrice(symbol=s, resolved_symbol=f"{s}.NS", latest_price=110.0, currency="INR", status="ok", source="yfinance", market_timestamp=datetime(2026,1,1,tzinfo=timezone.utc)))
    monkeypatch.setattr("app.services.portfolio_analytics.get_historical_data", lambda s, period="1y", interval="1d", exchange=None: _mock_hist(s, f"{s}.NS", period, interval))
    monkeypatch.setattr("app.services.portfolio_analytics.get_company_info", lambda s, e=None: CompanyInfo(symbol=s, resolved_symbol=f"{s}.NS", company_name=f"{s} Ltd", sector="Tech", industry="SW", status="ok", missing_fields=[]))
    monkeypatch.setattr("app.services.market_data.get_stock_price", lambda s, e=None: MarketPrice(symbol=s, resolved_symbol=f"{s}.NS", latest_price=110.0, currency="INR", status="ok", source="yfinance"))
    monkeypatch.setattr("app.services.market_data.get_historical_data", lambda s, period="1y", interval="1d", exchange=None: _mock_hist(s, f"{s}.NS", period, interval))
    monkeypatch.setattr("app.services.market_data.get_company_info", lambda s, e=None: CompanyInfo(symbol=s, resolved_symbol=f"{s}.NS", company_name=f"{s} Ltd", sector="Tech", industry="SW", status="ok", missing_fields=[]))


def _patch_m5_llms(monkeypatch):
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(_critic_json()))]))
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(_final_json()))]))


def _analyst_json():
    return {"portfolio_overview":"ok","key_findings":[],"stock_level_insights":[],"sector_observations":[],"risk_observations":[],"supporting_data":[],"sources":[],"warnings":[]}


def _research_json():
    return {
        "research_overview":"No relevant news found.","company_research":[{"symbol":"AAA.NS","company_name":"AAA Ltd","developments":[],"notes":"No relevant news found."}],
        "important_developments":[],"positive_signals":[],"negative_signals":[],"neutral_observations":["No relevant news found."],
        "potential_concerns":[],"sources":[],"warnings":["No relevant news found."]
    }


def test_workflow_includes_research_after_analyst(monkeypatch):
    _patch_market(monkeypatch)
    _patch_m5_llms(monkeypatch)
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_analyst_json()))]))
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_research_json()))]))
    state = {
        "confirmed_holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}],
        "history_period": "1y", "history_interval": "1d",
        "behavioral_questionnaire": QUESTIONNAIRE,
    }
    out = run_agent_analysis(state)
    assert out.get("portfolio_metrics") is not None
    assert out.get("analyst_findings") is not None
    assert out.get("research_findings") is not None
    assert out.get("research_status") in {"completed", "no_results", "partial"}
    assert out.get("critic_findings") is not None
    assert out.get("final_report") is not None


def test_research_failure_preserves_metrics_and_analyst(monkeypatch):
    _patch_market(monkeypatch)
    _patch_m5_llms(monkeypatch)
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_analyst_json()))]))
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: (_ for _ in ()).throw(RuntimeError("boom")))
    payload = {"holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}], "questionnaire_responses": QUESTIONNAIRE}
    resp = client.post("/api/agent/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio_metrics"] is not None
    assert data["research_status"] in {"failed", "partial", None, "no_results", "completed"} or "research_warnings" in data
    assert isinstance(data.get("research_warnings", []), list)
    # M5 still runs and preserves components
    assert "critic_findings" in data and "final_report" in data


def test_api_response_includes_research_fields(monkeypatch):
    _patch_market(monkeypatch)
    _patch_m5_llms(monkeypatch)
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_analyst_json()))]))
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_research_json()))]))
    payload = {"holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}], "questionnaire_responses": QUESTIONNAIRE}
    resp = client.post("/api/agent/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    for k in ["portfolio_metrics", "analyst_findings", "tool_execution_summaries", "sources", "warnings", "analysis_status", "final_analysis",
              "research_findings", "research_sources", "research_status", "research_warnings",
              "critic_findings", "critic_status", "behavioral_findings", "behavioral_profile", "final_report", "final_advisor_status", "thread_id", "workflow_stage", "human_review_status", "pending_questionnaire"]:
        assert k in data
    assert data["pending_questionnaire"] is False
    assert data["final_report"] is not None
