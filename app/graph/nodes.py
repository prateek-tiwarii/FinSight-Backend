import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from app.agents.tools import MARKET_ANALYST_TOOLS
from app.core.llm import build_chat_model, log_llm_response
from app.models.agent import MarketAnalystFindings
from app.models.analysis import ConfirmedHolding
from app.services.portfolio_analytics import analyze_portfolio

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are FinSight AI's Market Analyst Agent.

AUTHORITATIVE DATA & CONSTRAINTS:
1. Financial calculations supplied by Python/deterministic services are authoritative. Do not recalculate or override them.
2. Do not invent stock prices, financial metrics, or market events. Only state what is supported by provided data or tool results.
3. Call available tools (get_stock_price, get_historical_data, get_company_info) when additional evidence is necessary to support your analysis. Tools are read-only.
4. Distinguish clearly between observations (from data) and interpretations (your analysis).
5. Acknowledge missing or unavailable data explicitly.
6. Do not guarantee future returns.
7. Do not present unsupported buy/sell instructions.

OUTPUT REQUIREMENTS:
- When you have enough information and do NOT need to call tools, provide your final answer as a single JSON object (no markdown fences) matching this schema:
{
  "portfolio_overview": "string",
  "key_findings": ["string"],
  "stock_level_insights": [{"symbol": "...", "observation": "...", "interpretation": "..."}],
  "sector_observations": ["string"],
  "risk_observations": ["string"],
  "supporting_data": [{"description": "...", "value": ..., "source": "..."}],
  "sources": [{"name": "...", "type": "yfinance/calculation", "detail": "..."}],
  "warnings": ["string"]
}
- Be concise and evidence-based. Do not include hidden chain-of-thought.
- If tools fail or data is missing, reflect that in warnings and avoid unsupported claims."""


def _compact_market_data(metrics_dict: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for h in (metrics_dict.get("holdings") or []):
        sym = h.get("resolved_symbol") or h.get("input_symbol")
        hist = h.get("historical_data") or {}
        points = hist.get("points") or []
        closes = [p.get("close") for p in points if p.get("close") is not None]
        hist_summary = {
            "status": hist.get("status"),
            "period": hist.get("period"),
            "interval": hist.get("interval"),
            "observation_count": len(points),
            "message": hist.get("message"),
        }
        if closes:
            hist_summary.update({
                "date_start": points[0].get("date") if points else None,
                "date_end": points[-1].get("date") if points else None,
                "first_close": closes[0],
                "last_close": closes[-1],
                "min_close": min(closes),
                "max_close": max(closes),
            })
        compact[sym] = {
            "input_symbol": h.get("input_symbol"),
            "resolved_symbol": h.get("resolved_symbol"),
            "latest_price": h.get("latest_price"),
            "status": h.get("status"),
            "message": h.get("message"),
            "historical_summary": hist_summary,
        }
    return compact


def _compact_company_info(metrics_dict: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for h in (metrics_dict.get("holdings") or []):
        sym = h.get("resolved_symbol") or h.get("input_symbol")
        c = h.get("company") or {}
        info[sym] = {
            "company_name": c.get("company_name"),
            "sector": c.get("sector"),
            "industry": c.get("industry"),
            "market_cap": c.get("market_cap"),
            "pe_ratio": c.get("pe_ratio"),
            "dividend_yield": c.get("dividend_yield"),
            "status": c.get("status"),
            "missing_fields": c.get("missing_fields"),
            "message": c.get("message"),
        }
    return info


def portfolio_analysis_node(state: dict[str, Any]) -> dict[str, Any]:
    """Invoke existing deterministic portfolio analytics and populate shared state."""
    try:
        raw_holdings = state.get("confirmed_holdings") or []
        holdings = [ConfirmedHolding.model_validate(h) for h in raw_holdings]
        period = state.get("history_period") or "1y"
        interval = state.get("history_interval") or "1d"
        metrics = analyze_portfolio(holdings=holdings, history_period=period, history_interval=interval)
        metrics_dict = metrics.model_dump(mode="json")
        data_sources = [{
            "name": "yfinance",
            "type": "market_data",
            "detail": "Deterministic portfolio analytics and market data via yfinance",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
        warnings = list(metrics_dict.get("warnings") or [])
        return {
            "portfolio_metrics": metrics_dict,
            "market_data": _compact_market_data(metrics_dict),
            "company_information": _compact_company_info(metrics_dict),
            "data_sources": data_sources,
            "warnings": warnings,
            "analysis_status": "portfolio_ready",
        }
    except Exception as exc:
        logger.exception("portfolio_analysis_node failed")
        return {
            "analysis_status": "failed",
            "warnings": [f"Portfolio analysis failed: {exc}"],
        }


def _build_llm(with_tools: bool = True):
    llm = build_chat_model(temperature=0)
    if with_tools:
        return llm.bind_tools(MARKET_ANALYST_TOOLS)
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
    # try to find object
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


def _portfolio_context_for_prompt(state: dict[str, Any]) -> str:
    pm = state.get("portfolio_metrics") or {}
    summary = pm.get("summary") or {}
    risk = pm.get("risk") or {}
    holdings = pm.get("holdings") or []
    # compact holding view for prompt
    compact_holdings = []
    for h in holdings:
        compact_holdings.append({
            "input_symbol": h.get("input_symbol"),
            "resolved_symbol": h.get("resolved_symbol"),
            "quantity": h.get("quantity"),
            "average_purchase_price": h.get("average_purchase_price"),
            "invested_value": h.get("invested_value"),
            "latest_price": h.get("latest_price"),
            "current_value": h.get("current_value"),
            "absolute_pnl": h.get("absolute_pnl"),
            "percentage_return": h.get("percentage_return"),
            "allocation_percent": h.get("allocation_percent"),
            "sector": h.get("sector"),
            "status": h.get("status"),
            "message": h.get("message"),
            "company": {
                "company_name": (h.get("company") or {}).get("company_name"),
                "sector": (h.get("company") or {}).get("sector"),
                "industry": (h.get("company") or {}).get("industry"),
                "market_cap": (h.get("company") or {}).get("market_cap"),
                "pe_ratio": (h.get("company") or {}).get("pe_ratio"),
                "dividend_yield": (h.get("company") or {}).get("dividend_yield"),
                "missing_fields": (h.get("company") or {}).get("missing_fields"),
                "status": (h.get("company") or {}).get("status"),
            },
            "historical_summary": (state.get("market_data") or {}).get(h.get("resolved_symbol") or h.get("input_symbol"), {}).get("historical_summary"),
        })
    ctx = {
        "summary": summary,
        "risk": {
            "portfolio_volatility_annualized": risk.get("portfolio_volatility_annualized"),
            "portfolio_volatility_note": risk.get("portfolio_volatility_note"),
            "max_drawdown": risk.get("max_drawdown"),
            "max_drawdown_note": risk.get("max_drawdown_note"),
            "concentration_top_holding_percent": risk.get("concentration_top_holding_percent"),
            "concentration_hhi": risk.get("concentration_hhi"),
            "concentration_note": risk.get("concentration_note"),
            "assumptions": risk.get("assumptions"),
        },
        "holdings": compact_holdings,
        "warnings": pm.get("warnings") or [],
        "limitations": pm.get("limitations") or [],
        "data_source": pm.get("data_source"),
        "generated_at": pm.get("generated_at"),
    }
    return json.dumps(ctx, ensure_ascii=False)


def market_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
    """LLM-powered Market Analyst agent node with bounded tool use."""
    try:
        if state.get("analysis_status") == "failed":
            return {}

        iterations = int(state.get("tool_iterations") or 0)
        max_iters = int(state.get("max_tool_iterations") or 3)

        # collect summaries for tool results from latest messages
        msgs = list(state.get("agent_messages") or [])
        new_summaries: list[dict[str, Any]] = []
        for m in reversed(msgs):
            if isinstance(m, ToolMessage):
                # only recent batch; stop at first AI not tool? simple: summarize any ToolMessage not yet summarized by checking name+id in state is complex,
                # so we will summarize in tool node wrapper? easier: derive from messages added this turn by checking if already in summaries.
                pass
        # Instead, summarize tool messages that don't have matching summary ids
        existing_ids = {s.get("tool_call_id") for s in (state.get("tool_execution_summaries") or []) if s.get("tool_call_id")}
        tool_warnings: list[str] = []
        for m in msgs:
            if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) not in existing_ids:
                content = m.content
                status = "unknown"
                message = None
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                    if isinstance(parsed, dict):
                        status = parsed.get("status", "ok")
                        message = parsed.get("message")
                except Exception:
                    if isinstance(content, str) and content.lstrip().lower().startswith("error"):
                        status = "error"
                        message = content
                new_summaries.append({
                    "tool_call_id": getattr(m, "tool_call_id", None),
                    "tool_name": getattr(m, "name", None),
                    "status": status,
                })
                if status in {"error", "unresolved", "missing_data"}:
                    tool_name = getattr(m, "name", None) or "tool"
                    tool_warnings.append(
                        f"{tool_name} tool returned {status}: {message or 'no details available'}"
                    )

        llm = _build_llm(with_tools=True)
        if not msgs:
            context = _portfolio_context_for_prompt(state)
            msgs = [
                SystemMessage(content=ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=(
                    "Analyze the following authoritative portfolio metrics and context. "
                    "Use tools only if additional evidence is necessary. "
                    "When ready, return the final JSON analysis as specified.\n\n"
                    f"PORTFOLIO_CONTEXT_JSON:\n{context}"
                )),
            ]

        response = llm.invoke(msgs)
        log_llm_response("market_analyst", response)
        updates: dict[str, Any] = {"agent_messages": [response], "analysis_status": "analyzing"}
        if new_summaries:
            updates["tool_execution_summaries"] = new_summaries
        if tool_warnings:
            updates["warnings"] = tool_warnings

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            # If we've hit limit, do not route to tools; force finalize without executing more
            if iterations >= max_iters:
                warn = f"Maximum tool-call iterations reached ({max_iters}); proceeding without additional tool calls."
                updates.setdefault("warnings", []).append(warn)
                updates["analysis_status"] = "partial"
                # try to get structured result without tools
                try:
                    llm_no_tools = _build_llm(with_tools=False)
                    finalize_msgs = list(msgs) + [response, HumanMessage(content=(
                        "Tool-call limit reached. Do not request tools. Provide the best final JSON analysis now based on available information, "
                        "and include warnings about any missing evidence."
                    ))]
                    final_resp = llm_no_tools.invoke(finalize_msgs)
                    content = final_resp.content if isinstance(final_resp.content, str) else str(final_resp.content)
                    obj = _extract_json_object(content)
                    if obj:
                        findings = MarketAnalystFindings.model_validate(obj)
                        updates["analyst_findings"] = findings.model_dump()
                        updates["final_analysis"] = findings.model_dump()
                        updates["analysis_status"] = "completed"
                    else:
                        updates.setdefault("warnings", []).extend([warn, "Failed to produce structured final analysis after tool limit."])
                except Exception as exc:
                    logger.exception("Finalize after tool limit failed")
                    updates.setdefault("warnings", []).extend([warn, f"Failed to finalize analysis: {exc}"])
                return updates
            # normal tool-call path; increment iteration count for this agent->tools cycle
            updates["tool_iterations"] = iterations + 1
            return updates

        # No tool calls -> final answer expected
        content = response.content if isinstance(response.content, str) else str(response.content)
        obj = _extract_json_object(content)
        if obj is None:
            # try structured-output LLM as fallback
            try:
                llm_struct = _build_llm(with_tools=False).with_structured_output(MarketAnalystFindings)
                struct_msgs = list(msgs) + [response, HumanMessage(content="Return the final analysis as the required JSON object only.")]
                findings = llm_struct.invoke(struct_msgs)
                updates["analyst_findings"] = findings.model_dump()
                updates["final_analysis"] = findings.model_dump()
                updates["analysis_status"] = "completed"
                return updates
            except Exception as exc:
                logger.exception("Failed to parse/structure final analyst output")
                updates["analysis_status"] = "partial"
                updates.setdefault("warnings", []).append(f"Market Analyst returned unstructured/invalid output: {exc}")
                updates["final_analysis"] = {"raw_text": content[:4000]}
                return updates
        try:
            findings = MarketAnalystFindings.model_validate(obj)
            updates["analyst_findings"] = findings.model_dump()
            updates["final_analysis"] = findings.model_dump()
            updates["analysis_status"] = "completed"
        except ValidationError as exc:
            logger.exception("Market Analyst structured output validation failed")
            updates["analysis_status"] = "partial"
            updates.setdefault("warnings", []).append(f"Market Analyst output failed validation: {exc}")
            updates["final_analysis"] = {"raw": obj}
        except Exception as exc:
            logger.exception("Unexpected error processing analyst output")
            updates["analysis_status"] = "partial"
            updates.setdefault("warnings", []).append(f"Failed to process Market Analyst output: {exc}")
        return updates

    except Exception as exc:
        logger.exception("market_analyst_node failed")
        # Deterministic metrics may already exist; do not invalidate them
        return {
            "analysis_status": "partial" if state.get("portfolio_metrics") else "failed",
            "warnings": [f"Market Analyst agent failed: {exc}"],
        }
