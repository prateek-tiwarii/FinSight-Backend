import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from app.core.config import settings
from app.models.agent import RiskCriticFindings

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are FinSight AI's Risk Critic Agent.
You independently review and challenge existing analysis. Do not repeat conclusions; critically assess them.

INPUTS (authoritative where noted):
- Deterministic portfolio metrics (authoritative for numerical values) 
- Market Analyst findings
- Research Agent findings and sources

RESPONSIBILITIES:
- Identify unsupported claims, contradictions between market analysis and news research (only if real), concentration/sector exposure risks, significant negative developments, missing/stale/insufficient data, overconfident interpretations, overlooked risks, and claims without sufficient supporting evidence.
- Explain why each identified issue matters.
- Reference existing research sources when challenging news-related conclusions.

RULES:
1. Never invent financial metrics or events.
2. Use deterministic Python calculations as authoritative for numerical metrics; do not override them.
3. Do not manufacture contradictions when findings are consistent.
4. Do not issue unsupported buy/sell instructions.
5. Distinguish confirmed observations from potential risks.
6. If evidence is insufficient, explicitly state it.
7. Return concise evidence-based explanations only (no hidden chain-of-thought).

OUTPUT: Return ONLY a single JSON object (no markdown fences) matching:
{
  "critic_overview": "string",
  "challenged_claims": [{"claim":"string","challenged_by":"string|null","reasoning":"string|null","evidence":[{"...": "..."}]}],
  "identified_risks": [{"title":"string","description":"string","severity":"low|medium|high|critical","evidence":[{"...":"..."}],"related_stock_or_sector":"string|null","reasoning_summary":"string|null"}],
  "contradictions": [{"description":"string","market_analyst_point":"...","research_point":"...","evidence":"..."}],
  "missing_evidence": ["string"],
  "severity_classification": {"overall":"low|medium|high|critical|mixed","rationale":"string"},
  "recommendations_for_report": ["string"],
  "supporting_sources": [{"title":"...","url":"...","publisher":"...","publication_date":"..."}],
  "warnings": ["string"]
}
"""


def _build_llm():
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


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


def _context(state: dict[str, Any]) -> str:
    pm = state.get("portfolio_metrics") or {}
    analyst = state.get("analyst_findings") or {}
    research = state.get("research_findings") or {}
    ctx = {
        "deterministic_portfolio_metrics": {
            "summary": pm.get("summary"),
            "risk": pm.get("risk"),
            "holdings": [
                {
                    "symbol": h.get("resolved_symbol") or h.get("input_symbol"),
                    "allocation_percent": h.get("allocation_percent"),
                    "sector": h.get("sector"),
                    "status": h.get("status"),
                    "message": h.get("message"),
                    "percentage_return": h.get("percentage_return"),
                    "current_value": h.get("current_value"),
                }
                for h in (pm.get("holdings") or [])
            ],
            "warnings": pm.get("warnings"),
            "limitations": pm.get("limitations"),
            "generated_at": pm.get("generated_at"),
        },
        "market_analyst_findings": analyst,
        "research_findings": research,
        "research_sources": state.get("research_sources") or [],
        "existing_warnings": state.get("warnings") or [],
    }
    return json.dumps(ctx, ensure_ascii=False, default=str)


def risk_critic_node(state: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {"critic_status": "reviewing", "workflow_stage": "risk_critic"}
    try:
        # must run after analyst and research per workflow; be resilient to partial upstream
        if not state.get("portfolio_metrics") and not state.get("analyst_findings") and not state.get("research_findings"):
            return {
                "critic_status": "failed",
                "critic_findings": None,
                "warnings": ["Risk critic skipped: no upstream analysis available."],
                "workflow_stage": "risk_critic",
            }
        llm = _build_llm()
        msgs = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=(
                "Review and challenge the following analysis. Return the required JSON only.\n\n"
                f"CONTEXT_JSON:\n{_context(state)}"
            )),
        ]
        resp = llm.invoke(msgs)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        obj = _extract_json_object(content)
        if obj is None:
            try:
                findings = _build_llm().with_structured_output(RiskCriticFindings).invoke(
                    msgs + [HumanMessage(content="Return the final critic findings as the required JSON object only.")]
                )
                updates["critic_findings"] = findings.model_dump()
                updates["critic_status"] = "completed"
                return updates
            except Exception as exc:
                logger.exception("Critic structured fallback failed")
                updates["critic_status"] = "partial"
                updates["critic_findings"] = {
                    "critic_overview": "Critic output could not be structured.",
                    "challenged_claims": [], "identified_risks": [], "contradictions": [],
                    "missing_evidence": ["Critic output unstructured"], "severity_classification": {"overall": "unknown"},
                    "recommendations_for_report": [], "supporting_sources": [], "warnings": [str(exc)],
                }
                updates["warnings"] = [f"Risk critic returned unstructured output: {exc}"]
                return updates
        try:
            findings = RiskCriticFindings.model_validate(obj)
            updates["critic_findings"] = findings.model_dump()
            updates["critic_status"] = "completed"
        except ValidationError as exc:
            logger.exception("Critic validation failed")
            updates["critic_status"] = "partial"
            updates["critic_findings"] = {"raw": obj, "warnings": [str(exc)], "critic_overview": "Validation failed", "challenged_claims": [], "identified_risks": [], "contradictions": [], "missing_evidence": [], "severity_classification": {"overall": "unknown"}, "recommendations_for_report": [], "supporting_sources": []}
            updates["warnings"] = [f"Risk critic output failed validation: {exc}"]
        return updates
    except Exception as exc:
        logger.exception("risk_critic_node failed")
        return {
            "critic_status": "failed",
            "critic_findings": state.get("critic_findings"),
            "warnings": [f"Risk critic failed: {exc}"],
            "workflow_stage": "risk_critic",
        }
