import json

from langchain_core.messages import AIMessage

from app.agents.risk_critic import risk_critic_node
from app.models.agent import RiskCriticFindings


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


def _base_state():
    return {
        "portfolio_metrics": {
            "summary": {"total_current_value": 1000, "analyzed_holdings_count": 1},
            "risk": {"concentration_top_holding_percent": 100.0, "portfolio_volatility_annualized": 0.4},
            "holdings": [{"resolved_symbol": "AAA.NS", "allocation_percent": 100.0, "sector": "Tech", "status": "ok"}],
            "warnings": [],
            "limitations": [],
        },
        "analyst_findings": {"key_findings": ["Strong growth guaranteed"], "warnings": []},
        "research_findings": {"important_developments": [], "sources": [], "warnings": ["No news"]},
        "research_sources": [],
        "warnings": [],
    }


def test_risk_critic_valid_structured_findings(monkeypatch):
    payload = {
        "critic_overview": "Reviewed findings.",
        "challenged_claims": [{"claim": "Strong growth guaranteed", "challenged_by": "Risk Critic", "reasoning": "No guarantee supported by data.", "evidence": [{"source": "deterministic_metrics"}]}],
        "identified_risks": [{"title": "Concentration", "description": "Single holding dominates portfolio.", "severity": "high", "evidence": [{"metric": "concentration_top_holding_percent", "value": 100.0}], "related_stock_or_sector": "AAA.NS", "reasoning_summary": "Increases idiosyncratic risk."}],
        "contradictions": [],
        "missing_evidence": ["No supporting news evidence for growth claim"],
        "severity_classification": {"overall": "high", "rationale": "Concentration and unsupported claims."},
        "recommendations_for_report": ["Disclose concentration risk."],
        "supporting_sources": [],
        "warnings": [],
    }
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(payload))]))
    out = risk_critic_node(_base_state())
    assert out["critic_status"] == "completed"
    findings = RiskCriticFindings.model_validate(out["critic_findings"])
    assert findings.challenged_claims[0].claim == "Strong growth guaranteed"
    assert findings.identified_risks[0].severity == "high"


def test_risk_critic_detects_unsupported_claim_and_missing_evidence(monkeypatch):
    payload = {
        "critic_overview": "Unsupported claims identified.",
        "challenged_claims": [{"claim": "Will double in 6 months", "challenged_by": "Risk Critic", "reasoning": "No evidence supports guaranteed returns.", "evidence": []}],
        "identified_risks": [],
        "contradictions": [],
        "missing_evidence": ["Evidence for projected return claim"],
        "severity_classification": {"overall": "medium"},
        "recommendations_for_report": ["Remove unsupported projections."],
        "supporting_sources": [],
        "warnings": [],
    }
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(payload))]))
    out = risk_critic_node(_base_state())
    findings = out["critic_findings"]
    assert findings["missing_evidence"]
    assert any("double" in c["claim"].lower() or "unsupported" in c.get("reasoning", "").lower() for c in findings["challenged_claims"])


def test_risk_critic_contradictory_findings(monkeypatch):
    state = _base_state()
    state["analyst_findings"] = {"key_findings": ["Company outlook positive"], "warnings": []}
    state["research_findings"] = {"important_developments": [{"headline": "Regulatory probe announced", "sentiment": "negative", "source_url": "http://x/1"}], "sources": [{"url": "http://x/1"}], "warnings": []}
    payload = {
        "critic_overview": "Contradiction noted.",
        "challenged_claims": [],
        "identified_risks": [],
        "contradictions": [{"description": "Positive outlook conflicts with negative regulatory development.", "market_analyst_point": "Positive outlook", "research_point": "Regulatory probe announced", "evidence": "http://x/1"}],
        "missing_evidence": [],
        "severity_classification": {"overall": "medium"},
        "recommendations_for_report": ["Explain conflicting signals."],
        "supporting_sources": [{"url": "http://x/1"}],
        "warnings": [],
    }
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(payload))]))
    out = risk_critic_node(state)
    assert out["critic_status"] == "completed"
    assert out["critic_findings"]["contradictions"]


def test_risk_critic_no_contradiction_scenario(monkeypatch):
    payload = {
        "critic_overview": "Findings broadly consistent.",
        "challenged_claims": [],
        "identified_risks": [],
        "contradictions": [],
        "missing_evidence": [],
        "severity_classification": {"overall": "low"},
        "recommendations_for_report": ["Maintain disclosures."],
        "supporting_sources": [],
        "warnings": [],
    }
    monkeypatch.setattr("app.agents.risk_critic._build_llm", lambda: FakeLLM([AIMessage(content=json.dumps(payload))]))
    out = risk_critic_node(_base_state())
    assert out["critic_findings"]["contradictions"] == []
    assert out["critic_status"] == "completed"


def test_risk_critic_agent_failure_handling(monkeypatch):
    def boom():
        raise RuntimeError("LLM down")
    monkeypatch.setattr("app.agents.risk_critic._build_llm", boom)
    state = _base_state()
    out = risk_critic_node(state)
    assert out["critic_status"] == "failed"
    assert any("Risk critic failed" in w for w in out.get("warnings", []))
    # preserves prior findings if any
    state["critic_findings"] = {"critic_overview": "prior"}
    out2 = risk_critic_node(state)
    assert out2.get("critic_findings") == {"critic_overview": "prior"}
