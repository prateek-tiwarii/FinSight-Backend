import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class PortfolioState(TypedDict, total=False):
    """Shared LangGraph state for FinSight AI agentic analysis.

    Kept serializable; avoids duplicating large raw market datasets by storing
    summaries/metrics rather than full OHLCV where possible.
    """

    user_id: str | None
    confirmed_holdings: list[dict[str, Any]]
    history_period: str
    history_interval: str

    # Deterministic outputs (authoritative)
    portfolio_metrics: dict[str, Any] | None  # PortfolioAnalyzeResponse as dict
    market_data: dict[str, Any]  # compact per-symbol summaries (no large OHLCV)
    company_information: dict[str, Any]

    # Agentic
    agent_messages: Annotated[list[Any], add_messages]
    analyst_findings: dict[str, Any] | None

    # Research (M4)
    research_messages: Annotated[list[Any], add_messages]
    research_findings: dict[str, Any] | None
    retrieved_news: Annotated[list[dict[str, Any]], operator.add]
    research_sources: Annotated[list[dict[str, Any]], operator.add]
    research_warnings: Annotated[list[str], operator.add]
    research_status: str  # pending|researching|completed|partial|failed|no_results
    research_tool_iterations: int

    # Critic (M5)
    critic_findings: dict[str, Any] | None
    critic_status: str  # pending|reviewing|completed|partial|failed

    # Behavioral / HITL (M5)
    behavioral_questionnaire: dict[str, Any] | None
    behavioral_profile: dict[str, Any] | None
    behavioral_findings: dict[str, Any] | None
    behavioral_status: str  # pending|awaiting_input|completed|partial|failed
    human_review_status: str  # not_required|pending|completed
    pending_questionnaire: bool

    # Final advisor (M5)
    final_report: dict[str, Any] | None
    final_advisor_status: str  # pending|synthesizing|completed|partial|failed

    # Workflow
    workflow_stage: str

    # Cross-cutting
    identified_risks: Annotated[list[str], operator.add]
    data_sources: Annotated[list[dict[str, Any]], operator.add]
    warnings: Annotated[list[str], operator.add]
    tool_execution_summaries: Annotated[list[dict[str, Any]], operator.add]

    analysis_status: str  # pending|portfolio_ready|analyzing|completed|failed|partial|awaiting_user_input
    final_analysis: dict[str, Any] | None
    tool_iterations: int
    max_tool_iterations: int
