import pytest
from pydantic import ValidationError

from app.agents.behavioral_advisor import behavioral_advisor_node
from app.models.agent import (
    BehavioralFindings,
    BehavioralProfile,
    BehavioralQuestionnaireResponse,
    get_behavioral_questionnaire_schema,
)


def test_questionnaire_schema_and_validation():
    schema = get_behavioral_questionnaire_schema()
    keys = {q.key for q in schema.questions}
    assert {"investment_horizon", "risk_tolerance", "primary_investment_objective", "reaction_to_15_20pct_decline", "investment_experience"}.issubset(keys)
    # valid
    resp = BehavioralQuestionnaireResponse(
        investment_horizon="3_5_years",
        risk_tolerance="moderate",
        primary_investment_objective="long_term_wealth_creation",
        reaction_to_15_20pct_decline="hold",
        investment_experience="intermediate",
        additional_context={},
    )
    assert resp.risk_tolerance == "moderate"


def test_questionnaire_missing_required_answers():
    with pytest.raises(ValidationError):
        BehavioralQuestionnaireResponse(
            investment_horizon="3_5_years",
            risk_tolerance="moderate",
            # missing others
        )  # type: ignore


def _portfolio_metrics(concentrated=True, high_vol=True):
    return {
        "summary": {"sector_allocation": {"Tech": 100.0}, "analyzed_holdings_count": 1},
        "risk": {
            "concentration_top_holding_percent": 80.0 if concentrated else 20.0,
            "concentration_hhi": 6400.0 if concentrated else 2000.0,
            "portfolio_volatility_annualized": 0.45 if high_vol else 0.10,
            "max_drawdown": -0.30 if high_vol else -0.05,
        },
        "holdings": [{"resolved_symbol": "AAA.NS", "allocation_percent": 80.0 if concentrated else 20.0, "status": "ok"}],
    }


def test_behavioral_valid_profile_and_mismatch_conservative():
    q = {
        "investment_horizon": "less_than_1_year",
        "risk_tolerance": "conservative",
        "primary_investment_objective": "capital_preservation",
        "reaction_to_15_20pct_decline": "sell_most",
        "investment_experience": "beginner",
        "additional_context": {},
    }
    out = behavioral_advisor_node({"behavioral_questionnaire": q, "portfolio_metrics": _portfolio_metrics(True, True)})
    assert out["behavioral_status"] == "completed"
    assert out["human_review_status"] == "completed"
    assert out["pending_questionnaire"] is False
    profile = BehavioralProfile.model_validate(out["behavioral_profile"])
    findings = BehavioralFindings.model_validate(out["behavioral_findings"])
    assert profile.stated_preferences.risk_tolerance == "conservative"
    assert len(profile.mismatch_indicators) >= 1
    assert any(m.get("area") in {"risk_tolerance", "investment_horizon", "objective"} for m in profile.mismatch_indicators)
    assert findings.potential_behavioral_risks


def test_behavioral_different_horizons():
    base = {
        "risk_tolerance": "moderate",
        "primary_investment_objective": "long_term_wealth_creation",
        "reaction_to_15_20pct_decline": "hold",
        "investment_experience": "intermediate",
        "additional_context": {},
    }
    for horizon in ["less_than_1_year", "1_3_years", "3_5_years", "more_than_5_years"]:
        q = {**base, "investment_horizon": horizon}
        out = behavioral_advisor_node({"behavioral_questionnaire": q, "portfolio_metrics": _portfolio_metrics(True, True)})
        assert out["behavioral_status"] == "completed"
        assert out["behavioral_profile"]["stated_preferences"]["investment_horizon"] == horizon


def test_behavioral_requires_questionnaire():
    out = behavioral_advisor_node({"portfolio_metrics": _portfolio_metrics()})
    assert out["behavioral_status"] == "failed"
    assert any("without questionnaire" in w.lower() for w in out.get("warnings", []))
