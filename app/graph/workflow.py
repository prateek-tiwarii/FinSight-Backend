import logging
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from app.agents.behavioral_advisor import behavioral_advisor_node
from app.agents.final_advisor import final_advisor_node
from app.agents.research_agent import research_agent_node
from app.agents.risk_critic import risk_critic_node
from app.agents.tools import MARKET_ANALYST_TOOLS, RESEARCH_TOOLS
from app.graph.checkpointer import get_persistent_checkpointer
from app.graph.nodes import market_analyst_node, portfolio_analysis_node
from app.graph.state import PortfolioState

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS_DEFAULT = 3


def _route_after_analyst(state: PortfolioState) -> Literal["analyst_tools", "research"]:
    """Conditional edge: route to analyst tools if requested, else proceed to research."""
    if state.get("analysis_status") in {"failed"} and not state.get("portfolio_metrics"):
        return "research"
    messages = state.get("agent_messages") or []
    if not messages:
        return "research"
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        if state.get("analysis_status") in {"completed", "partial"} and (state.get("tool_iterations") or 0) >= (state.get("max_tool_iterations") or MAX_TOOL_ITERATIONS_DEFAULT):
            return "research"
        return "analyst_tools"
    return "research"


def _route_after_research(state: PortfolioState) -> Literal["research_tools", "critic"]:
    """Conditional edge: route to research tools if requested, else proceed to critic."""
    messages = state.get("research_messages") or []
    if not messages:
        return "critic"
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        if state.get("research_status") in {"completed", "partial", "failed", "no_results"} and (state.get("research_tool_iterations") or 0) >= (state.get("max_tool_iterations") or MAX_TOOL_ITERATIONS_DEFAULT):
            return "critic"
        return "research_tools"
    return "critic"


def _route_after_critic(state: PortfolioState) -> Literal["questionnaire_interrupt", "behavioral"]:
    # After critic, go to behavioral questionnaire/HITL; if already completed resume to behavioral
    if state.get("behavioral_status") == "completed" and state.get("behavioral_questionnaire"):
        return "behavioral"
    if state.get("human_review_status") == "completed" and state.get("behavioral_questionnaire"):
        return "behavioral"
    if state.get("pending_questionnaire") or state.get("human_review_status") == "pending":
        return "questionnaire_interrupt"
    # default: require questionnaire
    return "questionnaire_interrupt"


def _route_after_behavioral(state: PortfolioState) -> Literal["final_advisor"]:
    if state.get("behavioral_status") in {"completed"} and state.get("behavioral_questionnaire"):
        return "final_advisor"
    # even if failed, try to produce final report with available inputs
    return "final_advisor"


def questionnaire_node(state: PortfolioState):
    """HITL interrupt node: pause for user questionnaire responses."""
    # allow pre-supplied or completed questionnaire (backward compat / programmatic runs)
    if state.get("behavioral_questionnaire"):
        return {
            "human_review_status": "completed",
            "pending_questionnaire": False,
            "behavioral_status": state.get("behavioral_status") or "pending",
            "workflow_stage": "behavioral_ready",
        }
    if state.get("human_review_status") in {"completed", "skipped"}:
        return {
            "pending_questionnaire": False,
            "workflow_stage": "behavioral_ready",
        }
    payload = {
        "type": "questionnaire_required",
        "message": "Please complete the behavioral questionnaire to continue analysis.",
    }
    response = interrupt(payload)
    questionnaire = None
    if isinstance(response, dict):
        questionnaire = response.get("questionnaire_responses") or response.get("behavioral_questionnaire") or response
    elif hasattr(response, "questionnaire_responses"):
        questionnaire = getattr(response, "questionnaire_responses")
    if hasattr(questionnaire, "model_dump"):
        questionnaire = questionnaire.model_dump()
    if questionnaire and isinstance(questionnaire, dict):
        return {
            "behavioral_questionnaire": questionnaire,
            "human_review_status": "completed",
            "pending_questionnaire": False,
            "workflow_stage": "behavioral_ready",
        }
    # should not normally reach; treat as awaiting
    return {
        "human_review_status": "pending",
        "pending_questionnaire": True,
        "workflow_stage": "awaiting_user_input",
        "analysis_status": "awaiting_user_input",
    }


def build_agent_graph(checkpointer: Any = None):
    """Build and compile the FinSight AI LangGraph workflow (M3–M5)."""
    workflow = StateGraph(PortfolioState)

    workflow.add_node("portfolio_analysis", portfolio_analysis_node)
    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node(
        "analyst_tools",
        ToolNode(MARKET_ANALYST_TOOLS, messages_key="agent_messages", handle_tool_errors=True),
    )
    workflow.add_node("research", research_agent_node)
    workflow.add_node(
        "research_tools",
        ToolNode(RESEARCH_TOOLS, messages_key="research_messages", handle_tool_errors=True),
    )
    workflow.add_node("critic", risk_critic_node)
    workflow.add_node("questionnaire_interrupt", questionnaire_node)
    workflow.add_node("behavioral", behavioral_advisor_node)
    workflow.add_node("final_advisor", final_advisor_node)

    workflow.add_edge(START, "portfolio_analysis")
    workflow.add_edge("portfolio_analysis", "market_analyst")
    workflow.add_conditional_edges(
        "market_analyst",
        _route_after_analyst,
        {"analyst_tools": "analyst_tools", "research": "research"},
    )
    workflow.add_edge("analyst_tools", "market_analyst")
    workflow.add_conditional_edges(
        "research",
        _route_after_research,
        {"research_tools": "research_tools", "critic": "critic"},
    )
    workflow.add_edge("research_tools", "research")
    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"questionnaire_interrupt": "questionnaire_interrupt", "behavioral": "behavioral"},
    )
    # after interrupt/resume questionnaire_node returns; proceed to behavioral
    workflow.add_edge("questionnaire_interrupt", "behavioral")
    workflow.add_conditional_edges(
        "behavioral",
        _route_after_behavioral,
        {"final_advisor": "final_advisor"},
    )
    workflow.add_edge("final_advisor", END)

    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


_graph_app = None
_checkpointer = None


def get_agent_graph(checkpointer: Any = None):
    global _graph_app, _checkpointer
    if checkpointer is not None:
        return build_agent_graph(checkpointer)
    persistent = get_persistent_checkpointer()
    if _graph_app is None or _checkpointer is not persistent:
        _checkpointer = persistent
        _graph_app = build_agent_graph(_checkpointer)
    return _graph_app


def get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = get_persistent_checkpointer()
    return _checkpointer


def reset_agent_graph() -> None:
    """Clear the compiled graph cache (used after checkpointer resets/tests)."""
    global _graph_app, _checkpointer
    _graph_app = None
    _checkpointer = None


def run_agent_analysis(initial_state: dict[str, Any], thread_id: str | None = None) -> dict[str, Any]:
    """Execute compiled graph with provided initial state and return final/snapshot state."""
    app = get_agent_graph()
    # ensure defaults
    if "tool_iterations" not in initial_state:
        initial_state["tool_iterations"] = 0
    if "research_tool_iterations" not in initial_state:
        initial_state["research_tool_iterations"] = 0
    if "max_tool_iterations" not in initial_state:
        initial_state["max_tool_iterations"] = MAX_TOOL_ITERATIONS_DEFAULT
    if "analysis_status" not in initial_state:
        initial_state["analysis_status"] = "pending"
    if "research_status" not in initial_state:
        initial_state["research_status"] = "pending"
    if "critic_status" not in initial_state:
        initial_state["critic_status"] = "pending"
    if "behavioral_status" not in initial_state:
        initial_state["behavioral_status"] = "pending"
    if "human_review_status" not in initial_state:
        initial_state["human_review_status"] = "pending"
    # if questionnaire pre-supplied, do not require pending HITL
    if initial_state.get("behavioral_questionnaire"):
        initial_state.setdefault("human_review_status", "completed")
        initial_state.setdefault("pending_questionnaire", False)
    if "pending_questionnaire" not in initial_state:
        initial_state["pending_questionnaire"] = not bool(initial_state.get("behavioral_questionnaire"))
    if "final_advisor_status" not in initial_state:
        initial_state["final_advisor_status"] = "pending"
    if "workflow_stage" not in initial_state:
        initial_state["workflow_stage"] = "started"
    try:
        tid = thread_id or initial_state.get("thread_id") or initial_state.get("user_id") or "default-thread"
        config = {"configurable": {"thread_id": tid}}
        result = app.invoke(initial_state, config=config)
        if isinstance(result, dict):
            result.setdefault("thread_id", tid)
            # if interrupted, invoke may return state without explicit flags in some cases; infer
            snapshot = app.get_state(config)
            interrupts = getattr(snapshot, "interrupts", None) or ()
            if interrupts:
                result["pending_questionnaire"] = True
                result["human_review_status"] = "pending"
                result["workflow_stage"] = "awaiting_user_input"
                result["analysis_status"] = "awaiting_user_input"
        return result
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise


def get_analysis_state(thread_id: str) -> dict[str, Any] | None:
    app = get_agent_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = app.get_state(config)
    except Exception:
        return None
    if snapshot is None:
        return None
    values = getattr(snapshot, "values", None) or {}
    if not values:
        return None
    result = dict(values)
    result.setdefault("thread_id", thread_id)
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if interrupts:
        result["pending_questionnaire"] = True
        result["human_review_status"] = "pending"
        result["workflow_stage"] = "awaiting_user_input"
        result["analysis_status"] = "awaiting_user_input"
    return result


def resume_agent_analysis(thread_id: str, questionnaire_responses: dict[str, Any]) -> dict[str, Any]:
    app = get_agent_graph()
    config = {"configurable": {"thread_id": thread_id}}
    # ensure thread exists / has pending interrupt
    state = get_analysis_state(thread_id)
    if state is None:
        raise ValueError("Invalid or unknown thread_id")
    try:
        result = app.invoke(Command(resume={"questionnaire_responses": questionnaire_responses}), config=config)
        if isinstance(result, dict):
            result.setdefault("thread_id", thread_id)
        return result
    except Exception as exc:
        logger.exception("Graph resume failed")
        raise

