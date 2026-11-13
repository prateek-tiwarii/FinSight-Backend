import json

from langchain_core.messages import AIMessage

from app.agents.final_advisor import final_advisor_node
from app.models.agent import FinalPortfolioReport


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, msgs):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    def with_structured_output(self, schema):
        return self


def _state():
    return {
        "portfolio_metrics": {
            "summary": {"total_invested_amount": 1000, "total_current_value": 1100, "total_absolute_pnl": 100, "overall_percentage_return": 10.0},
            "risk": {"concentration_top_holding_percent": 100.0},
            "holdings": [{"resolved_symbol": "AAA.NS", "sector": "Tech", "allocation_percent": 100.0, "status": "ok", "percentage_return": 10.0}],
            "warnings": [],
            "limitations": [],
        },
        "analyst_findings": {"portfolio_overview": "Up 10%", "key_findings": ["Positive return"]},
        "research_findings": {"research_overview": "No major news", "important_developments": []},
        "research_sources": [{"title": "Article", "url": "http://example.com/a", "publisher": "Pub", "publication_date": "2026-01-01"}],
        "critic_findings": {"critic_overview": "Concentration risk", "identified_risks": [{"title": "Concentration", "severity": "high"}], "contradictions": []},
        "behavioral_profile": {"stated_preferences": {"risk_tolerance": "moderate"}},
        "behavioral_findings": {"behavioral_overview": "Some mismatch", "areas_of_conflict": [{"type": "risk_mismatch"}]},
        "analysis_status": "completed",
        "research_status": "completed",
        "critic_status": "completed",
        "behavioral_status": "completed",
        "warnings": [],
        "research_warnings": [],
    }


def _report_json():
    return {
        "report_title": "FinSight AI Portfolio Intelligence Report",
        "executive_summary": "Portfolio value 1100 with 10% return (deterministic).",
        "overall_assessment": "Positive performance with concentration considerations.",
        "portfolio_strengths": ["Positive return"],
        "portfolio_weaknesses": ["Concentration"],
        "performance_insights": [{"area": "returns", "insight": "Overall +10%", "evidence": "summary.overall_percentage_return=10.0"}],
        "risk_assessment": {"summary": "Concentration flagged by critic", "key_risks": ["Concentration"], "critic_highlights": ["High severity concentration"]},
        "research_summary": {"overview": "No major news", "important_developments": [], "sources": [{"url": "http://example.com/a"}]},
        "behavioral_assessment": {"overview": "Moderate tolerance with mismatch noted", "profile_summary": {"risk_tolerance": "moderate"}, "conflicts": [{"type": "risk_mismatch"}]},
        "preference_mismatches": [{"area": "risk", "detail": "mismatch"}],
        "key_observations": ["Deterministic metrics show +10% return."],
        "areas_requiring_attention": ["Concentration risk"],
        "critic_findings_summary": {"overview": "Concentration risk", "severity": "high", "top_concerns": ["Concentration"]},
        "supporting_sources": [{"title": "Article", "url": "http://example.com/a", "publisher": "Pub", "publication_date": "2026-01-01"}],
        "warnings": [],
        "generated_at": "2026-01-01T00:00:00Z",
    }


def test_final_advisor_synthesizes_and_includes_all(monkeypatch):
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(_report_json()))]))
    out = final_advisor_node(_state())
    assert out["final_advisor_status"] == "completed"
    assert out["analysis_status"] == "completed"
    report = FinalPortfolioReport.model_validate(out["final_report"])
    assert "10" in report.executive_summary
    assert report.critic_findings_summary.get("overview")
    assert report.behavioral_assessment.get("overview")
    assert any(s.get("url") == "http://example.com/a" for s in report.supporting_sources)
    assert report.risk_assessment


def test_final_advisor_preserves_sources_and_deterministic_refs(monkeypatch):
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(_report_json()))]))
    out = final_advisor_node(_state())
    report = out["final_report"]
    # source preserved
    assert any("example.com" in (s.get("url") or "") for s in report.get("supporting_sources", []))
    # deterministic metric value referenced
    assert "10" in json.dumps(report)


def test_final_advisor_handles_missing_failed_agents(monkeypatch):
    # LLM fails -> fallback partial report should still include available components/warnings
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    state = _state()
    state["research_findings"] = None
    state["critic_status"] = "failed"
    state["behavioral_status"] = "failed"
    state["warnings"] = ["upstream issue"]
    out = final_advisor_node(state)
    assert out["final_advisor_status"] in {"partial", "failed"}
    report = out["final_report"]
    assert report is not None
    assert report.get("report_title")
    assert any("partial" in w.lower() or "failed" in w.lower() or "boom" in w for w in (report.get("warnings") or []))
    # still references available metrics
    assert report.get("performance_insights")


def test_final_advisor_does_not_require_invention_when_data_missing(monkeypatch):
    # ensure fallback doesn't fabricate numbers by checking it carries through known summary
    monkeypatch.setattr("app.agents.final_advisor._build_llm", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = final_advisor_node(_state())
    report = out["final_report"]
    assert report["performance_insights"][0]["evidence"]["overall_percentage_return"] == 10.0
