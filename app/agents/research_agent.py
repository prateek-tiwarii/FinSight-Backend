import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from app.agents.tools import RESEARCH_TOOLS
from app.core.llm import build_chat_model, log_llm_response
from app.models.agent import ResearchFindings

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are FinSight AI's Research Agent.

SCOPE & CONSTRAINTS:
1. Never fabricate news, events, dates, financial figures, or sources. Only use retrieved evidence (tool results) and provided context.
2. Never present speculation as fact. Distinguish observations (from retrieved articles/data) from interpretations.
3. Do not assume a headline proves a company's financial performance.
4. Do not treat old news as recent; respect publication dates and lookback context.
5. Do not claim an article was verified beyond retrieved RSS metadata.
6. If no relevant news is found for a holding/company, explicitly report that.
7. Avoid unsupported buy/sell recommendations and do not guarantee future stock performance.
8. Cite supporting articles (URL, source, publication date when available) for every material claim/development.
9. Do not force sentiment classification when evidence is ambiguous (use "unclear" or "mixed" and explain).

TOOL USE:
- Use search_company_news (and get_company_info if helpful) to gather evidence. Tools are read-only.
- Research each relevant portfolio holding/company without unnecessarily repeating identical queries.
- Keep findings concise and evidence-backed.

OUTPUT:
- When ready (no more tool calls needed), return ONLY a single JSON object (no markdown fences) matching:
{
  "research_overview": "string",
  "company_research": [
    {
      "symbol": "string|null",
      "company_name": "string|null",
      "developments": [<development objects>],
      "notes": "string|null"
    }
  ],
  "important_developments": [<development objects>],
  "positive_signals": ["string"],
  "negative_signals": ["string"],
  "neutral_observations": ["string"],
  "potential_concerns": ["string"],
  "sources": [{"title":"...","url":"...","publisher":"...","publication_date":"...","related_company":"..."}],
  "warnings": ["string"]
}

Each development object:
{
  "related_company": "string|null",
  "headline": "string|null",
  "summary": "string|null",
  "sentiment": "positive|negative|neutral|mixed|unclear",
  "publication_date": "string|null",
  "source_name": "string|null",
  "source_url": "string|null",
  "why_it_may_matter": "string|null",
  "evidence_or_limitation_notes": "string|null"
}
If evidence is insufficient, reflect that in warnings/notes and avoid unsupported claims."""


def _build_llm(with_tools: bool = True):
    llm = build_chat_model(temperature=0)
    if with_tools:
        return llm.bind_tools(RESEARCH_TOOLS)
    return llm


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


def _research_context(state: dict[str, Any]) -> str:
    holdings = state.get("confirmed_holdings") or []
    pm = state.get("portfolio_metrics") or {}
    metric_holdings = pm.get("holdings") or []
    company_info = state.get("company_information") or {}
    # build compact per-holding research targets
    targets = []
    seen = set()
    for h in metric_holdings:
        sym = h.get("resolved_symbol") or h.get("input_symbol")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        c = h.get("company") or company_info.get(sym) or {}
        targets.append({
            "symbol": sym,
            "input_symbol": h.get("input_symbol"),
            "company_name": c.get("company_name"),
            "sector": c.get("sector"),
            "industry": c.get("industry"),
        })
    # fallback to confirmed holdings if metrics missing
    if not targets:
        for h in holdings:
            sym = h.get("stock_symbol")
            if not sym or sym in seen:
                continue
            seen.add(sym)
            targets.append({"symbol": sym, "input_symbol": sym, "company_name": None})

    analyst = state.get("analyst_findings") or {}
    ctx = {
        "holdings_to_research": targets,
        "analyst_key_findings": analyst.get("key_findings") if isinstance(analyst, dict) else [],
        "analyst_warnings": analyst.get("warnings") if isinstance(analyst, dict) else [],
    }
    return json.dumps(ctx, ensure_ascii=False)


def _collect_tool_outputs(messages: list[Any], existing_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    summaries = []
    retrieved = []
    sources = []
    warnings = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        tid = getattr(m, "tool_call_id", None)
        if tid in existing_ids:
            continue
        name = getattr(m, "name", None)
        content = m.content
        parsed = None
        status = "unknown"
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            status = parsed.get("status", "ok")
            # capture news articles as retrieved evidence
            if name == "search_company_news":
                arts = parsed.get("articles") or []
                if isinstance(arts, list):
                    for a in arts[:10]:
                        if isinstance(a, dict):
                            retrieved.append(a)
                            if a.get("url") or a.get("title"):
                                sources.append({
                                    "title": a.get("title"),
                                    "url": a.get("url"),
                                    "publisher": a.get("publisher"),
                                    "publication_date": a.get("publication_date"),
                                    "related_company": a.get("related_company") or parsed.get("company_name") or parsed.get("symbol"),
                                    "source": "google_news_rss",
                                })
                for w in (parsed.get("warnings") or []):
                    if isinstance(w, str):
                        warnings.append(w)
                if parsed.get("message") and status in {"error", "no_results"}:
                    warnings.append(f"{parsed.get('symbol') or parsed.get('query')}: {parsed.get('message')}")
        summaries.append({"tool_call_id": tid, "tool_name": name, "status": status})
    return summaries, retrieved, sources, warnings


def research_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Research agent node: gathers news evidence via tools and produces structured findings."""
    updates: dict[str, Any] = {}
    try:
        # do not block on analyst failure if we have holdings context; but be careful
        if state.get("analysis_status") == "failed" and not state.get("portfolio_metrics"):
            return {
                "research_status": "failed",
                "research_warnings": ["Research skipped because portfolio analysis failed."],
            }

        iterations = int(state.get("research_tool_iterations") or 0)
        max_iters = int(state.get("max_tool_iterations") or 3)
        msgs = list(state.get("research_messages") or [])

        existing_ids = {s.get("tool_call_id") for s in (state.get("tool_execution_summaries") or []) if s.get("tool_call_id")}
        new_summaries, new_retrieved, new_sources, new_warns = _collect_tool_outputs(msgs, existing_ids)
        if new_summaries:
            updates["tool_execution_summaries"] = new_summaries
        if new_retrieved:
            updates["retrieved_news"] = new_retrieved
        if new_sources:
            updates["research_sources"] = new_sources
        if new_warns:
            updates["research_warnings"] = new_warns

        llm = _build_llm(with_tools=True)
        if not msgs:
            ctx = _research_context(state)
            msgs = [
                SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                HumanMessage(content=(
                    "Research the following portfolio holdings using available tools. "
                    "Gather relevant evidence (news/company info) and return the final JSON findings as specified.\n\n"
                    f"RESEARCH_CONTEXT_JSON:\n{ctx}"
                )),
            ]

        response = llm.invoke(msgs)
        log_llm_response("research_agent", response)
        updates["research_messages"] = [response]
        updates.setdefault("research_status", "researching")

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            if iterations >= max_iters:
                warn = f"Maximum research tool-call iterations reached ({max_iters}); finalizing with available evidence."
                updates.setdefault("research_warnings", [])
                updates["research_warnings"] = updates["research_warnings"] + [warn]
                try:
                    llm_no_tools = _build_llm(with_tools=False)
                    finalize_msgs = list(msgs) + [response, HumanMessage(content=(
                        "Tool-call limit reached. Do not request tools. Provide the best final research JSON now based on available evidence only. "
                        "If no relevant news was found, state that explicitly and include warnings."
                    ))]
                    final_resp = llm_no_tools.invoke(finalize_msgs)
                    content = final_resp.content if isinstance(final_resp.content, str) else str(final_resp.content)
                    obj = _extract_json_object(content)
                    if obj:
                        findings = ResearchFindings.model_validate(obj)
                        updates["research_findings"] = findings.model_dump()
                        updates["research_status"] = "partial" if (findings.warnings or warn) else "completed"
                        return updates
                except Exception as exc:
                    logger.exception("Research finalize after tool limit failed")
                    updates["research_warnings"] = updates.get("research_warnings", []) + [f"Failed to finalize research: {exc}"]
                updates["research_status"] = "partial"
                return updates
            updates["research_tool_iterations"] = iterations + 1
            updates["research_status"] = "researching"
            return updates

        # no tool calls -> final
        content = response.content if isinstance(response.content, str) else str(response.content)
        obj = _extract_json_object(content)
        if obj is None:
            try:
                llm_struct = _build_llm(with_tools=False).with_structured_output(ResearchFindings)
                struct_msgs = list(msgs) + [response, HumanMessage(content="Return the final research findings as the required JSON object only.")]
                findings = llm_struct.invoke(struct_msgs)
                updates["research_findings"] = findings.model_dump()
                updates["research_status"] = "completed"
                return updates
            except Exception as exc:
                logger.exception("Failed to structure research output")
                updates["research_status"] = "partial"
                updates["research_warnings"] = updates.get("research_warnings", []) + [f"Research agent returned unstructured/invalid output: {exc}"]
                updates["research_findings"] = {
                    "research_overview": "Research output could not be structured.",
                    "company_research": [],
                    "important_developments": [],
                    "positive_signals": [],
                    "negative_signals": [],
                    "neutral_observations": [],
                    "potential_concerns": [],
                    "sources": [],
                    "warnings": [f"Unstructured output: {str(exc)}"],
                }
                return updates
        try:
            findings = ResearchFindings.model_validate(obj)
            status = "completed"
            if not findings.important_developments and not findings.company_research:
                status = "no_results"
            updates["research_findings"] = findings.model_dump()
            updates["research_status"] = status
        except ValidationError as exc:
            logger.exception("Research output validation failed")
            updates["research_status"] = "partial"
            updates["research_warnings"] = updates.get("research_warnings", []) + [f"Research findings failed validation: {exc}"]
            updates["research_findings"] = {"raw": obj, "warnings": [str(exc)]}
        except Exception as exc:
            logger.exception("Unexpected research processing error")
            updates["research_status"] = "partial"
            updates["research_warnings"] = updates.get("research_warnings", []) + [f"Failed to process research findings: {exc}"]
        return updates

    except Exception as exc:
        logger.exception("research_agent_node failed")
        return {
            "research_status": "failed",
            "research_warnings": [f"Research agent failed: {exc}"],
            "research_findings": state.get("research_findings"),
        }
