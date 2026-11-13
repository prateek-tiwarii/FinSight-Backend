import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.core.llm import build_chat_model, log_llm_response
from app.models.agent import FinalPortfolioReport

logger = logging.getLogger(__name__)

FINAL_SYSTEM_PROMPT = """You are FinSight AI's Final Advisor Agent. Synthesize all available findings into one comprehensive portfolio intelligence report.

INPUTS (use all that are available):
- Deterministic portfolio metrics (AUTHORITATIVE for all numerical financial values)
- Market Analyst findings
- Research Agent findings and sources
- Risk Critic findings
- Behavioral profile and findings

RULES:
1. Use actual Python-calculated metrics; do not invent or alter financial figures.
2. Preserve evidence and source URLs from research/inputs.
3. Distinguish facts/observations from interpretations.
4. Reflect the Risk Critic's challenges and concerns.
5. Consider the user's questionnaire responses/profile.
6. Clearly disclose missing, stale, or unavailable data and any failed agent components.
7. No guaranteed-return claims. No unsupported buy/sell recommendations.
8. Never fabricate financial figures or research findings.
9. If components are missing/failed, produce a partial report and disclose what is missing.

OUTPUT: Return ONLY a single JSON object (no markdown fences) matching:
{
  "report_title": "string",
  "executive_summary": "string",
  "overall_assessment": "string",
  "portfolio_strengths": ["string"],
  "portfolio_weaknesses": ["string"],
  "performance_insights": [{"area":"...","insight":"...","evidence":"..."}],
  "risk_assessment": {"summary":"...","key_risks":[...],"critic_highlights":[...]},
  "research_summary": {"overview":"...","important_developments":[...],"sources":[...]},
  "behavioral_assessment": {"overview":"...","profile_summary":{...},"conflicts":[...]},
  "preference_mismatches": [{"area":"...","detail":"..."}],
  "key_observations": ["string"],
  "areas_requiring_attention": ["string"],
  "critic_findings_summary": {"overview":"...","severity":"...","top_concerns":[...]},
  "supporting_sources": [{"title":"...","url":"...","publisher":"...","publication_date":"..."}],
  "warnings": ["string"],
  "generated_at": "ISO-8601 string"
}
"""


def _build_llm():
    return build_chat_model(temperature=0)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        import re
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    import re
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _synthesis_context(state: dict[str, Any]) -> str:
    pm = state.get("portfolio_metrics") or {}
    ctx = {
        "deterministic_portfolio_metrics": {
            "summary": pm.get("summary"),
            "risk": pm.get("risk"),
            "holdings": [
                {
                    "symbol": h.get("resolved_symbol") or h.get("input_symbol"),
                    "company_name": (h.get("company") or {}).get("company_name"),
                    "sector": h.get("sector"),
                    "quantity": h.get("quantity"),
                    "invested_value": h.get("invested_value"),
                    "current_value": h.get("current_value"),
                    "absolute_pnl": h.get("absolute_pnl"),
                    "percentage_return": h.get("percentage_return"),
                    "allocation_percent": h.get("allocation_percent"),
                    "status": h.get("status"),
                }
                for h in (pm.get("holdings") or [])
            ],
            "warnings": pm.get("warnings"),
            "limitations": pm.get("limitations"),
            "generated_at": pm.get("generated_at"),
            "data_source": pm.get("data_source"),
        },
        "market_analyst_findings": state.get("analyst_findings"),
        "research_findings": state.get("research_findings"),
        "research_sources": state.get("research_sources") or [],
        "critic_findings": state.get("critic_findings"),
        "behavioral_profile": state.get("behavioral_profile"),
        "behavioral_findings": state.get("behavioral_findings"),
        "component_statuses": {
            "analysis_status": state.get("analysis_status"),
            "research_status": state.get("research_status"),
            "critic_status": state.get("critic_status"),
            "behavioral_status": state.get("behavioral_status"),
        },
        "warnings": state.get("warnings") or [],
        "research_warnings": state.get("research_warnings") or [],
    }
    return json.dumps(ctx, ensure_ascii=False, default=str)


def _fallback_report(state: dict[str, Any], reason: str) -> dict[str, Any]:
    pm = state.get("portfolio_metrics") or {}
    summary = pm.get("summary") or {}
    research_sources = list(state.get("research_sources") or [])
    critic = state.get("critic_findings") or {}
    behavioral = state.get("behavioral_findings") or {}
    warnings = list(state.get("warnings") or []) + list(state.get("research_warnings") or [])
    warnings.append(reason)
    return {
        "report_title": "FinSight AI Portfolio Intelligence Report (Partial)",
        "executive_summary": "A partial report was generated because final synthesis could not be fully completed. Deterministic metrics and available findings are summarized where possible.",
        "overall_assessment": "Partial assessment due to synthesis limitations; review component warnings and available findings.",
        "portfolio_strengths": [],
        "portfolio_weaknesses": [],
        "performance_insights": [{"area": "summary", "insight": "Deterministic portfolio metrics available." if summary else "Deterministic metrics unavailable.", "evidence": summary}],
        "risk_assessment": {"summary": "See critic findings and deterministic risk metrics where available.", "critic_highlights": (critic.get("identified_risks") if isinstance(critic, dict) else [])},
        "research_summary": {"overview": ((state.get("research_findings") or {}).get("research_overview") if isinstance(state.get("research_findings"), dict) else None), "sources": research_sources},
        "behavioral_assessment": {"overview": (behavioral.get("behavioral_overview") if isinstance(behavioral, dict) else None)},
        "preference_mismatches": ((behavioral.get("areas_of_conflict") if isinstance(behavioral, dict) else []) or []),
        "key_observations": ["Report generated in fallback/partial mode."],
        "areas_requiring_attention": ["Review missing/failed components and warnings."],
        "critic_findings_summary": {"overview": (critic.get("critic_overview") if isinstance(critic, dict) else None)},
        "supporting_sources": research_sources,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def final_advisor_node(state: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {"final_advisor_status": "synthesizing", "workflow_stage": "final_advisor"}
    try:
        # must run after critic and behavioral advisor (workflow enforces), but handle partial inputs
        ctx = _synthesis_context(state)
        llm = _build_llm()
        msgs = [
            SystemMessage(content=FINAL_SYSTEM_PROMPT),
            HumanMessage(content=f"Synthesize a comprehensive report from the following context. Return required JSON only.\n\nCONTEXT_JSON:\n{ctx}"),
        ]
        resp = llm.invoke(msgs)
        log_llm_response("final_advisor", resp)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        obj = _extract_json_object(content)
        if obj is None:
            try:
                report = llm.with_structured_output(FinalPortfolioReport).invoke(
                    msgs + [HumanMessage(content="Return the final report as the required JSON object only.")]
                )
                updates["final_report"] = report.model_dump()
                updates["final_advisor_status"] = "completed"
                updates["workflow_stage"] = "completed"
                updates["analysis_status"] = "completed"
                return updates
            except Exception as exc:
                logger.exception("Final advisor structured fallback failed")
                updates["final_report"] = _fallback_report(state, f"Final synthesis unstructured/failed: {exc}")
                updates["final_advisor_status"] = "partial"
                updates["warnings"] = [f"Final advisor produced partial fallback report: {exc}"]
                updates["workflow_stage"] = "completed"
                updates["analysis_status"] = "partial"
                return updates
        try:
            report = FinalPortfolioReport.model_validate(obj)
            updates["final_report"] = report.model_dump()
            updates["final_advisor_status"] = "completed"
            updates["workflow_stage"] = "completed"
            updates["analysis_status"] = "completed"
        except ValidationError as exc:
            logger.exception("Final report validation failed")
            updates["final_report"] = _fallback_report(state, f"Final report validation failed: {exc}")
            updates["final_advisor_status"] = "partial"
            updates["warnings"] = [f"Final report validation failed; partial report returned: {exc}"]
            updates["workflow_stage"] = "completed"
            updates["analysis_status"] = "partial"
        return updates
    except Exception as exc:
        logger.exception("final_advisor_node failed")
        return {
            "final_advisor_status": "failed",
            "final_report": _fallback_report(state, f"Final advisor failed: {exc}"),
            "warnings": [f"Final advisor failed: {exc}"],
            "workflow_stage": "completed",
            "analysis_status": "partial" if state.get("portfolio_metrics") else "failed",
        }
