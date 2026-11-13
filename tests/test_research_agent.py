import json

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.research_agent import research_agent_node
from app.models.agent import ResearchFindings


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


def _base_state():
    return {
        "portfolio_metrics": {
            "holdings": [
                {
                    "input_symbol": "AAA",
                    "resolved_symbol": "AAA.NS",
                    "status": "ok",
                    "company": {"company_name": "AAA Ltd", "sector": "Tech", "industry": "SW", "status": "ok"},
                }
            ],
            "summary": {},
            "risk": {},
        },
        "confirmed_holdings": [{"stock_symbol": "AAA", "quantity": 1, "average_purchase_price": 100}],
        "company_information": {"AAA.NS": {"company_name": "AAA Ltd"}},
        "analyst_findings": {"key_findings": ["Profitable"]},
        "research_messages": [],
        "tool_execution_summaries": [],
        "research_tool_iterations": 0,
        "max_tool_iterations": 3,
        "analysis_status": "completed",
    }


def test_research_agent_structured_output_validation(monkeypatch):
    findings = {
        "research_overview": "Overview",
        "company_research": [{"symbol": "AAA.NS", "company_name": "AAA Ltd", "developments": [], "notes": "No relevant news found."}],
        "important_developments": [],
        "positive_signals": [],
        "negative_signals": [],
        "neutral_observations": ["No relevant news found."],
        "potential_concerns": [],
        "sources": [],
        "warnings": ["No relevant news found."],
    }
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(findings))]))
    out = research_agent_node(_base_state())
    assert out["research_status"] in {"completed", "no_results"}
    ResearchFindings.model_validate(out["research_findings"])


def test_research_agent_positive_negative_neutral_ambiguous(monkeypatch):
    findings = {
        "research_overview": "Mixed evidence.",
        "company_research": [],
        "important_developments": [
            {"related_company": "AAA Ltd", "headline": "Record profit", "summary": "Reported higher profit.", "sentiment": "positive", "publication_date": "2026-01-01", "source_name": "Pub", "source_url": "http://a/1", "why_it_may_matter": "May improve outlook.", "evidence_or_limitation_notes": "Based on RSS metadata only."},
            {"related_company": "AAA Ltd", "headline": "Regulatory probe", "summary": "Probe announced.", "sentiment": "negative", "publication_date": "2026-01-02", "source_name": "Pub", "source_url": "http://a/2", "why_it_may_matter": "Potential compliance risk.", "evidence_or_limitation_notes": "Details limited."},
            {"related_company": "AAA Ltd", "headline": "Routine AGM", "summary": "AGM held.", "sentiment": "neutral", "publication_date": "2026-01-03", "source_name": "Pub", "source_url": "http://a/3", "why_it_may_matter": "Routine governance.", "evidence_or_limitation_notes": None},
            {"related_company": "AAA Ltd", "headline": "Unclear impact", "summary": "Mixed signals.", "sentiment": "unclear", "publication_date": "2026-01-04", "source_name": "Pub", "source_url": "http://a/4", "why_it_may_matter": "Impact uncertain.", "evidence_or_limitation_notes": "Ambiguous evidence."},
        ],
        "positive_signals": ["Profit report (see cited article)."],
        "negative_signals": ["Regulatory probe (see cited article)."],
        "neutral_observations": ["Routine AGM."],
        "potential_concerns": ["Probe details unclear."],
        "sources": [{"title": "Record profit", "url": "http://a/1", "publisher": "Pub", "publication_date": "2026-01-01"}],
        "warnings": [],
    }
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(findings))]))
    out = research_agent_node(_base_state())
    rf = ResearchFindings.model_validate(out["research_findings"])
    sentiments = {d.sentiment for d in rf.important_developments}
    assert {"positive", "negative", "neutral", "unclear"}.issubset(sentiments)
    assert rf.sources[0]["url"] == "http://a/1"


def test_research_agent_no_news_scenario(monkeypatch):
    findings = {
        "research_overview": "No relevant recent news was found for the portfolio holdings in the searched period.",
        "company_research": [{"symbol": "AAA.NS", "company_name": "AAA Ltd", "developments": [], "notes": "No relevant news found."}],
        "important_developments": [],
        "positive_signals": [],
        "negative_signals": [],
        "neutral_observations": [],
        "potential_concerns": [],
        "sources": [],
        "warnings": ["No relevant news found for AAA.NS in the last 30 days."],
    }
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(findings))]))
    out = research_agent_node(_base_state())
    assert out["research_status"] in {"completed", "no_results"}
    assert "No relevant" in out["research_findings"]["research_overview"] or out["research_findings"]["warnings"]


def test_research_agent_tool_failure_handling(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr("app.agents.research_agent._build_llm", boom)
    out = research_agent_node(_base_state())
    assert out["research_status"] == "failed"
    assert out.get("research_warnings")


def test_research_agent_missing_source_metadata(monkeypatch):
    findings = {
        "research_overview": "Evidence retrieved but some metadata missing.",
        "company_research": [],
        "important_developments": [
            {"related_company": "AAA Ltd", "headline": "Update", "summary": "Some update.", "sentiment": "neutral", "publication_date": None, "source_name": None, "source_url": None, "why_it_may_matter": "Limited context.", "evidence_or_limitation_notes": "Source metadata unavailable; based on retrieved RSS metadata only."}
        ],
        "positive_signals": [],
        "negative_signals": [],
        "neutral_observations": ["Metadata incomplete."],
        "potential_concerns": [],
        "sources": [],
        "warnings": ["Some source metadata missing."],
    }
    monkeypatch.setattr("app.agents.research_agent._build_llm", lambda with_tools=True: FakeLLM([AIMessage(content=json.dumps(findings))]))
    out = research_agent_node(_base_state())
    dev = out["research_findings"]["important_developments"][0]
    assert dev["publication_date"] is None
    assert dev["source_url"] is None
    assert "metadata" in (dev.get("evidence_or_limitation_notes") or "").lower()
