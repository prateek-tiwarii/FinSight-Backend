import logging
from typing import Any

from app.models.agent import BehavioralFindings, BehavioralProfile, BehavioralQuestionnaireResponse

logger = logging.getLogger(__name__)


def _portfolio_profile_from_metrics(portfolio_metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not portfolio_metrics:
        return {"available": False, "note": "Deterministic portfolio metrics unavailable."}
    summary = portfolio_metrics.get("summary") or {}
    risk = portfolio_metrics.get("risk") or {}
    holdings = portfolio_metrics.get("holdings") or []
    top_alloc = risk.get("concentration_top_holding_percent")
    hhi = risk.get("concentration_hhi")
    vol = risk.get("portfolio_volatility_annualized")
    mdd = risk.get("max_drawdown")
    sector_alloc = summary.get("sector_allocation") or {}
    top_sector = None
    top_sector_pct = None
    if sector_alloc:
        try:
            top_sector, top_sector_pct = max(sector_alloc.items(), key=lambda x: float(x[1] or 0))
        except Exception:
            pass
    ok_holdings = [h for h in holdings if h.get("status") == "ok" and h.get("allocation_percent") is not None]
    concentrated = (top_alloc is not None and float(top_alloc) >= 35) or len(ok_holdings) <= 2
    high_vol = vol is not None and float(vol) >= 0.30
    large_drawdown = mdd is not None and float(mdd) <= -0.20
    return {
        "available": True,
        "top_holding_percent": top_alloc,
        "hhi": hhi,
        "volatility_annualized": vol,
        "max_drawdown": mdd,
        "sector_allocation": sector_alloc,
        "top_sector": top_sector,
        "top_sector_percent": top_sector_pct,
        "holdings_count": len(holdings),
        "analyzed_holdings_count": summary.get("analyzed_holdings_count"),
        "flags": {
            "concentrated": concentrated,
            "high_volatility": high_vol,
            "large_historical_drawdown": large_drawdown,
            "high_sector_exposure": (top_sector_pct is not None and float(top_sector_pct) >= 50),
        },
    }


def _build_behavioral_analysis(questionnaire: dict[str, Any], portfolio_metrics: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    qr = BehavioralQuestionnaireResponse.model_validate(questionnaire)
    profile_info = _portfolio_profile_from_metrics(portfolio_metrics)
    flags = (profile_info.get("flags") or {}) if profile_info.get("available") else {}

    mismatches: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    behavioral_risks: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    observations: list[str] = []
    educational: list[str] = []
    warnings: list[str] = []

    # Risk tolerance vs portfolio
    if profile_info.get("available"):
        comparisons.append({
            "aspect": "risk_tolerance_vs_portfolio",
            "stated": qr.risk_tolerance,
            "portfolio_indicators": {
                "concentrated": flags.get("concentrated"),
                "high_volatility": flags.get("high_volatility"),
                "large_historical_drawdown": flags.get("large_historical_drawdown"),
                "top_holding_percent": profile_info.get("top_holding_percent"),
                "volatility_annualized": profile_info.get("volatility_annualized"),
            },
            "note": "Comparison based on deterministic metrics and concentration/volatility indicators (educational, not a suitability determination).",
        })
        if qr.risk_tolerance == "conservative" and (flags.get("concentrated") or flags.get("high_volatility") or flags.get("large_historical_drawdown")):
            mismatches.append({
                "area": "risk_tolerance",
                "stated": "conservative",
                "observed": "portfolio shows concentration/volatility/drawdown characteristics that may be higher than a conservative preference",
                "evidence": {"flags": flags, "top_holding_percent": profile_info.get("top_holding_percent")},
            })
            conflicts.append({"type": "risk_mismatch", "description": "Conservative risk tolerance may conflict with observed portfolio risk characteristics."})
        if qr.risk_tolerance == "aggressive" and not (flags.get("concentrated") or flags.get("high_volatility")):
            mismatches.append({
                "area": "risk_tolerance",
                "stated": "aggressive",
                "observed": "portfolio does not exhibit strong concentration/volatility signals in available metrics",
                "evidence": {"flags": flags},
            })
    else:
        warnings.append("Portfolio metrics unavailable; unable to compare stated preferences with actual portfolio characteristics.")

    # Horizon vs characteristics
    comparisons.append({
        "aspect": "investment_horizon",
        "stated": qr.investment_horizon,
        "note": "Shorter horizons can be more sensitive to volatility and drawdowns; comparison is educational.",
    })
    if qr.investment_horizon in {"less_than_1_year", "1_3_years"} and (flags.get("high_volatility") or flags.get("large_historical_drawdown") or flags.get("concentrated")):
        mismatches.append({
            "area": "investment_horizon",
            "stated": qr.investment_horizon,
            "observed": "portfolio shows volatility/drawdown/concentration signals that may be challenging for shorter horizons",
            "evidence": {"flags": flags},
        })
        conflicts.append({"type": "horizon_mismatch", "description": "Shorter investment horizon may conflict with observed portfolio volatility/concentration."})

    # Reaction to decline
    reaction = qr.reaction_to_15_20pct_decline
    if reaction in {"sell_most", "sell_some"}:
        behavioral_risks.append({
            "behavior": "selling_during_drawdown",
            "stated_reaction": reaction,
            "risk": "Selling during significant declines can lock in losses and may conflict with long-term objectives if fundamentals are unchanged.",
            "evidence_basis": "User-stated preference (educational observation).",
        })
    elif reaction == "invest_more_if_fundamentals_strong":
        behavioral_risks.append({
            "behavior": "adding_during_drawdown",
            "stated_reaction": reaction,
            "risk": "Adding during declines requires liquidity, discipline, and evidence-based assessment of fundamentals; outcomes are uncertain.",
            "evidence_basis": "User-stated preference (educational observation).",
        })
    else:
        behavioral_risks.append({
            "behavior": "holding_during_drawdown",
            "stated_reaction": reaction,
            "risk": "Holding through declines can align with long-term plans but may be difficult if losses exceed risk tolerance or liquidity needs arise.",
            "evidence_basis": "User-stated preference (educational observation).",
        })

    # Objective considerations
    comparisons.append({
        "aspect": "primary_investment_objective",
        "stated": qr.primary_investment_objective,
        "note": "Objective alignment depends on holdings, income characteristics, and time horizon; deterministic income/dividend suitability not fully assessed here.",
    })
    if qr.primary_investment_objective == "capital_preservation" and (flags.get("high_volatility") or flags.get("concentrated")):
        mismatches.append({
            "area": "objective",
            "stated": qr.primary_investment_objective,
            "observed": "portfolio risk indicators may be inconsistent with capital preservation emphasis",
            "evidence": {"flags": flags},
        })

    observations.append("Behavioral profile based on user-provided questionnaire responses; not a professional psychometric assessment.")
    if not mismatches:
        observations.append("No strong preference-portfolio mismatches detected from available deterministic indicators.")
    educational.extend([
        "Consider aligning position sizing and diversification with stated risk tolerance and time horizon.",
        "Maintain awareness of liquidity needs before reacting to market declines.",
        "Use evidence (company fundamentals/news context) and deterministic metrics rather than headlines alone when reassessing holdings.",
    ])

    profile = BehavioralProfile(
        stated_preferences=qr,
        portfolio_risk_profile=profile_info,
        mismatch_indicators=mismatches,
        behavioral_observations=observations,
        educational_notes=educational,
        warnings=warnings,
    )
    findings = BehavioralFindings(
        behavioral_overview="Behavioral assessment compares stated preferences with deterministic portfolio characteristics. Educational observations only; not definitive suitability judgments.",
        profile=profile,
        comparisons_to_portfolio=comparisons,
        potential_behavioral_risks=behavioral_risks,
        areas_of_conflict=conflicts,
        recommendations_educational=educational,
        warnings=warnings,
    )
    return profile.model_dump(), findings.model_dump()


def behavioral_advisor_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        questionnaire = state.get("behavioral_questionnaire")
        if not questionnaire:
            return {
                "behavioral_status": "failed",
                "warnings": ["Behavioral advisor ran without questionnaire responses."],
                "workflow_stage": "behavioral_advisor",
            }
        profile, findings = _build_behavioral_analysis(questionnaire, state.get("portfolio_metrics"))
        return {
            "behavioral_profile": profile,
            "behavioral_findings": findings,
            "behavioral_status": "completed",
            "human_review_status": "completed",
            "pending_questionnaire": False,
            "workflow_stage": "behavioral_advisor",
        }
    except Exception as exc:
        logger.exception("behavioral_advisor_node failed")
        return {
            "behavioral_status": "failed",
            "warnings": [f"Behavioral advisor failed: {exc}"],
            "workflow_stage": "behavioral_advisor",
            "human_review_status": "completed" if state.get("behavioral_questionnaire") else "pending",
        }
