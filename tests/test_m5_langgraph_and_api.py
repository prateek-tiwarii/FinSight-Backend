import json
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from app.graph.workflow import build_agent_graph, get_agent_graph, get_analysis_state, resume_agent_analysis, run_agent_analysis
from app.models.analysis import CompanyInfo, HistoricalData, MarketPrice, OHLCVPoint
from tests.helpers import make_guest_client

client = make_guest_client()


class FakeLLM:
    def __init__(self, responses, counter=None):
        self.responses = list(responses)
        self.calls = 0
        self.counter = counter

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        if self.counter is not None:
            self.counter["calls"] += 1
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


def _patch_market(monkeypatch):
    monkeypatch.setattr("app.services.portfolio_analytics.get_stock_price", lambda s, e=None: MarketPrice(symbol=s, resolved_symbol=f"{s}.NS", latest_price=110.0, currency="INR", status="ok", source="yfinance", market_timestamp=datetime(2026,1,1,tzinfo=timezone.utc)))
    monkeypatch.setattr("app.services.portfolio_analytics.get_historical_data", lambda s, period="1y", interval="1d", exchange=None: _mock_hist(s, f"{s}.NS", period, interval))
    monkeypatch.setattr("app.services.portfolio_analytics.get_company_info", lambda s, e=None: CompanyInfo(symbol=s, resolved_symbol=f"{s}.NS", company_name=f"{s} Ltd", sector="Tech", industry="SW", status="ok", missing_fields=[]))
    monkeypatch.setattr("app.services.market_data.get_stock_price", lambda s, e=None: MarketPrice(symbol=s, resolved_symbol=f"{s}.NS", latest_price=110.0, currency="INR", status="ok", source="yfinance"))
    monkeypatch.setattr("app.services.market_data.get_historical_data", lambda s, period="1y", interval="1d", exchange=None: _mock_hist(s, f"{s}.NS", period, interval))
    monkeypatch.setattr("app.services.market_data.get_company_info", lambda s, e=None: CompanyInfo(symbol=s, resolved_symbol=f"{s}.NS", company_name=f"{s} Ltd", sector="Tech", industry="SW", status="ok", missing_fields=[]))


def _analyst_json():
    return {"portfolio_overview":"ok","key_findings":["k"],"stock_level_insights":[],"sector_observations":[],"risk_observations":[],"supporting_data":[],"sources":[],"warnings":[]}


def _research_json():
    return {"research_overview":"No relevant news found.","company_research":[],"important_developments":[],"positive_signals":[],"negative_signals":[],"neutral_observations":["No relevant news found."],"potential_concerns":[],"sources":[],"warnings":[]}


def _critic_json():
    return {"critic_overview":"ok","challenged_claims":[],"identified_risks":[{"title":"Concentration","description":"desc","severity":"medium","evidence":[],"related_stock_or_sector":"AAA.NS","reasoning_summary":"r"}],"contradictions":[],"missing_evidence":[],"severity_classification":{"overall":"medium"},"recommendations_for_report":[],"supporting_sources":[],"warnings":[]}


def _final_json():
    return {"report_title":"Report","executive_summary":"Exec","overall_assessment":"Overall","portfolio_strengths":[],"portfolio_weaknesses":[],"performance_insights":[],"risk_assessment":{"summary":"r"},"research_summary":{"overview":"r"},"behavioral_assessment":{"overview":"b"},"preference_mismatches":[],"key_observations":[],"areas_requiring_attention":[],"critic_findings_summary":{"overview":"c"},"supporting_sources":[],"warnings":[],"generated_at":"2026-01-01T00:00:00Z"}


QUESTIONNAIRE = {
    "investment_horizon": "3_5_years",
    "risk_tolerance": "moderate",
    "primary_investment_objective": "long_term_wealth_creation",
    "reaction_to_15_20pct_decline": "hold",
    "investment_experience": "intermediate",
    "additional_context": {"financial_goals": "retirement"},
}


def _patch_llms(monkeypatch, counters=None):
    counters = counters or {}
    a = counters.setdefault("analyst", {"calls": 0})
    r = counters.setdefault("research", {"calls": 0})
    c = counters.setdefault("critic", {"calls": 0})
    f = counters.setdefault("final", {"calls": 0})
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_analyst_json()))], a))
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_research_json()))], r))
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(_critic_json()))], c))
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(_final_json()))], f))
    return counters


def test_get_questionnaire_endpoint():
    resp = client.get("/api/agent/questionnaire")
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"]
    keys = {q["key"] for q in data["questions"]}
    assert "investment_horizon" in keys and "risk_tolerance" in keys


def test_initial_analysis_pauses_for_user_input_and_resume(monkeypatch):
    _patch_market(monkeypatch)
    counters = _patch_llms(monkeypatch)
    payload = {"holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}]}
    resp = client.post("/api/agent/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["thread_id"]
    assert data["pending_questionnaire"] is True
    assert data["human_review_status"] == "pending"
    assert data["analysis_status"] == "awaiting_user_input"
    assert data["workflow_stage"] == "awaiting_user_input"
    assert data["questionnaire_schema"] is not None
    # completed pre-HITL components preserved
    assert data["portfolio_metrics"] is not None
    assert data["analyst_findings"] is not None
    assert data["research_findings"] is not None
    assert data["critic_findings"] is not None
    assert data["final_report"] is None

    # resume
    thread_id = data["thread_id"]
    resp2 = client.post("/api/agent/resume", json={"thread_id": thread_id, "questionnaire_responses": QUESTIONNAIRE})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["pending_questionnaire"] is False
    assert data2["human_review_status"] == "completed"
    assert data2["behavioral_findings"] is not None
    assert data2["behavioral_profile"] is not None
    assert data2["final_report"] is not None
    assert data2["analysis_status"] == "completed"
    # questionnaire responses preserved
    assert data2["behavioral_profile"]["stated_preferences"]["investment_horizon"] == "3_5_years"

    # no duplicate execution of pre-interrupt LLM stages after resume
    assert counters["analyst"]["calls"] == 1
    assert counters["research"]["calls"] == 1
    assert counters["critic"]["calls"] == 1
    assert counters["final"]["calls"] == 1


def test_resume_invalid_thread_id():
    resp = client.post("/api/agent/resume", json={"thread_id": "nonexistent-thread", "questionnaire_responses": QUESTIONNAIRE})
    assert resp.status_code == 404


def test_resume_invalid_questionnaire_response(monkeypatch):
    _patch_market(monkeypatch)
    _patch_llms(monkeypatch)
    payload = {"holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}]}
    resp = client.post("/api/agent/analyze", json=payload)
    assert resp.status_code == 200
    tid = resp.json()["thread_id"]
    bad = {"investment_horizon": "3_5_years", "risk_tolerance": "moderate"}  # missing required
    resp2 = client.post("/api/agent/resume", json={"thread_id": tid, "questionnaire_responses": bad})
    assert resp2.status_code == 422


def test_graph_order_and_final_after_required_inputs(monkeypatch):
    _patch_market(monkeypatch)
    counters = _patch_llms(monkeypatch)
    # programmatic run with questionnaire supplied -> full flow to final
    state = {
        "confirmed_holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}],
        "history_period": "1y", "history_interval": "1d",
        "behavioral_questionnaire": QUESTIONNAIRE,
    }
    out = run_agent_analysis(state, thread_id=str(uuid.uuid4()))
    assert out.get("portfolio_metrics") is not None
    assert out.get("analyst_findings") is not None
    assert out.get("research_findings") is not None
    assert out.get("critic_findings") is not None
    assert out.get("behavioral_findings") is not None
    assert out.get("final_report") is not None
    assert out.get("final_advisor_status") == "completed"
    # all stages invoked
    assert counters["analyst"]["calls"] >= 1
    assert counters["research"]["calls"] >= 1
    assert counters["critic"]["calls"] >= 1
    assert counters["final"]["calls"] >= 1


def test_partial_failure_handling_preserves_earlier_findings(monkeypatch):
    _patch_market(monkeypatch)
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_analyst_json()))]))
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(_research_json()))]))
    # critic fails
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: (_ for _ in ()).throw(RuntimeError("critic down")))
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(_final_json()))]))
    state = {
        "confirmed_holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}],
        "behavioral_questionnaire": QUESTIONNAIRE,
    }
    out = run_agent_analysis(state, thread_id=str(uuid.uuid4()))
    assert out.get("portfolio_metrics") is not None
    assert out.get("analyst_findings") is not None
    assert out.get("research_findings") is not None
    assert out.get("critic_status") == "failed"
    assert out.get("behavioral_findings") is not None
    assert out.get("final_report") is not None  # synthesized with available inputs
