import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.dependencies import require_principal
from app.core.config import settings
from app.core.llm import (
    LLMAuthError,
    LLMBillingError,
    LLMModelUnavailableError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    get_llm_info,
)
from app.db.session import get_db
from app.graph.workflow import get_analysis_state, resume_agent_analysis, run_agent_analysis
from app.models.agent import (
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    AgentResumeRequest,
    BehavioralFindings,
    BehavioralProfile,
    BehavioralQuestionnaireResponse,
    BehavioralQuestionnaireSchema,
    FinalPortfolioReport,
    MarketAnalystFindings,
    ResearchFindings,
    RiskCriticFindings,
    get_behavioral_questionnaire_schema,
)
from app.models.analysis import PortfolioAnalyzeRequest, PortfolioAnalyzeResponse
from app.models.portfolio import PortfolioExtractionResponse
from app.services.auth_service import Principal
from app.services.extraction import ExtractionError, PortfolioExtractionService
from app.services.persistence import (
    GuestLimitExceeded,
    NotFoundError,
    create_analysis_run,
    ensure_analysis_portfolio,
    find_run_by_thread,
    persist_analysis_result,
)
from app.services.portfolio_analytics import analyze_portfolio

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


def _llm_http_status(error: LLMProviderError) -> tuple[int, str]:
    """Map a normalized LLM provider error to an honest HTTP status/detail."""
    if isinstance(error, LLMRateLimitError):
        return 429, "The AI provider is rate-limited right now. Please retry in a few moments."
    if isinstance(error, LLMTimeoutError):
        return 504, "The AI provider did not respond in time. Please retry."
    if isinstance(error, (LLMAuthError, LLMBillingError)):
        return 502, "The AI provider is not correctly configured (credentials or billing)."
    if isinstance(error, LLMModelUnavailableError):
        return 503, "The configured AI model is temporarily unavailable. Please retry shortly."
    return 502, "The AI provider failed to process the request."


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/extract/portfolio", response_model=PortfolioExtractionResponse)
async def extract_portfolio(file: UploadFile = File(...)) -> PortfolioExtractionResponse:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PNG, JPEG, WEBP.",
        )

    contents = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {settings.max_upload_size_mb} MB.",
        )

    service = PortfolioExtractionService()
    try:
        result = await service.extract(contents, file.content_type or "image/png")
    except LLMProviderError as exc:
        status, detail = _llm_http_status(exc)
        logger.warning(
            "Portfolio extraction provider error (%s, provider=%s model=%s): %s",
            status,
            get_llm_info().provider,
            get_llm_info().model,
            exc,
        )
        raise HTTPException(status_code=status, detail=detail) from exc
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected portfolio extraction failure")
        raise HTTPException(status_code=502, detail="Failed to extract portfolio from image.")

    return result


@router.post("/portfolio/analyze", response_model=PortfolioAnalyzeResponse)
async def analyze_portfolio_endpoint(payload: PortfolioAnalyzeRequest) -> PortfolioAnalyzeResponse:
    # Deterministic analytics; no LLM calls.
    return analyze_portfolio(
        holdings=payload.holdings,
        history_period=payload.history_period,
        history_interval=payload.history_interval,
    )


_NON_TERMINAL_STATUSES = {
    None,
    "",
    "pending",
    "researching",
    "reviewing",
    "synthesizing",
    "analyzing",
    "portfolio_ready",
}
# "no_results" is a valid research outcome, not a failure.
_SUCCESS_EQUIVALENT = {"completed", "no_results"}


def _derive_analysis_status(
    final_state: dict,
    findings_obj: object | None,
    final_report_obj: object | None,
    pending: bool,
) -> str:
    """Aggregate stage statuses into a single completed/partial/failed status.

    Deterministic metrics are preserved even when AI stages fail; those runs are
    reported as partial (not completed) so the UI never overstates reliability.
    """
    if pending:
        return "awaiting_user_input"
    if not final_state.get("portfolio_metrics"):
        return "failed"

    stages = [
        final_state.get("analysis_status"),
        final_state.get("research_status"),
        final_state.get("critic_status"),
        final_state.get("behavioral_status"),
        final_state.get("final_advisor_status"),
    ]
    terminal = [s for s in stages if s not in _NON_TERMINAL_STATUSES]
    if any(s not in _SUCCESS_EQUIVALENT for s in terminal):
        return "partial"

    if final_report_obj is not None and final_state.get("final_advisor_status") == "completed":
        return "completed"
    if findings_obj is not None and final_state.get("analysis_status") == "completed":
        return "completed"
    return "partial"


def _build_agent_response(final_state: dict, thread_id: str | None = None) -> AgentAnalyzeResponse:
    pm = final_state.get("portfolio_metrics")
    metrics_obj = PortfolioAnalyzeResponse.model_validate(pm) if pm else None

    findings = final_state.get("analyst_findings")
    findings_obj = MarketAnalystFindings.model_validate(findings) if findings else None

    research_findings = final_state.get("research_findings")
    research_obj = ResearchFindings.model_validate(research_findings) if research_findings else None

    critic = final_state.get("critic_findings")
    critic_obj = RiskCriticFindings.model_validate(critic) if critic else None

    behavioral_profile = final_state.get("behavioral_profile")
    behavioral_profile_obj = BehavioralProfile.model_validate(behavioral_profile) if behavioral_profile else None
    behavioral_findings = final_state.get("behavioral_findings")
    behavioral_findings_obj = BehavioralFindings.model_validate(behavioral_findings) if behavioral_findings else None

    final_report = final_state.get("final_report")
    final_report_obj = FinalPortfolioReport.model_validate(final_report) if final_report else None

    sources = list(final_state.get("data_sources") or [])
    seen = {str(s) for s in sources}
    for obj in [findings_obj, research_obj]:
        if obj and getattr(obj, "sources", None):
            for s in obj.sources:
                if str(s) not in seen:
                    sources.append(s)
                    seen.add(str(s))
    if critic_obj and critic_obj.supporting_sources:
        for s in critic_obj.supporting_sources:
            if str(s) not in seen:
                sources.append(s)
                seen.add(str(s))

    research_sources = list(final_state.get("research_sources") or [])
    if research_obj and research_obj.sources:
        for s in research_obj.sources:
            if str(s) not in {str(rs) for rs in research_sources}:
                research_sources.append(s)

    warnings = list(final_state.get("warnings") or [])
    if findings_obj and findings_obj.warnings:
        for w in findings_obj.warnings:
            if w not in warnings:
                warnings.append(w)
    if critic_obj and critic_obj.warnings:
        for w in critic_obj.warnings:
            if w not in warnings:
                warnings.append(w)
    if behavioral_findings_obj and behavioral_findings_obj.warnings:
        for w in behavioral_findings_obj.warnings:
            if w not in warnings:
                warnings.append(w)
    if final_report_obj and final_report_obj.warnings:
        for w in final_report_obj.warnings:
            if w not in warnings:
                warnings.append(w)

    research_warnings = list(final_state.get("research_warnings") or [])
    if research_obj and research_obj.warnings:
        for w in research_obj.warnings:
            if w not in research_warnings:
                research_warnings.append(w)

    # merge supporting sources from final report into top-level where possible
    if final_report_obj and final_report_obj.supporting_sources:
        for s in final_report_obj.supporting_sources:
            if str(s) not in seen:
                sources.append(s)
                seen.add(str(s))

    research_status = final_state.get("research_status")
    pending = final_state.get("pending_questionnaire")
    human_status = final_state.get("human_review_status")
    # ensure HITL consistency
    if pending is None:
        pending = human_status == "pending"
    pending = bool(pending)
    status = _derive_analysis_status(final_state, findings_obj, final_report_obj, pending)
    q_schema = get_behavioral_questionnaire_schema() if pending else None

    return AgentAnalyzeResponse(
        portfolio_metrics=metrics_obj,
        analyst_findings=findings_obj,
        research_findings=research_obj,
        research_sources=research_sources,
        research_status=research_status,
        research_warnings=research_warnings,
        tool_execution_summaries=list(final_state.get("tool_execution_summaries") or []),
        sources=sources,
        warnings=warnings,
        analysis_status=status,
        final_analysis=final_state.get("final_analysis"),
        thread_id=thread_id or final_state.get("thread_id"),
        workflow_stage=final_state.get("workflow_stage"),
        human_review_status=human_status,
        pending_questionnaire=pending,
        questionnaire_schema=q_schema,
        critic_findings=critic_obj,
        critic_status=final_state.get("critic_status"),
        behavioral_profile=behavioral_profile_obj,
        behavioral_findings=behavioral_findings_obj,
        behavioral_status=final_state.get("behavioral_status"),
        final_report=final_report_obj,
        final_advisor_status=final_state.get("final_advisor_status"),
    )


@router.get("/agent/questionnaire", response_model=BehavioralQuestionnaireSchema)
async def get_agent_questionnaire() -> BehavioralQuestionnaireSchema:
    return get_behavioral_questionnaire_schema()


def _persist_response(
    db: Session,
    principal: Principal,
    response: AgentAnalyzeResponse,
    run,
    questionnaire_responses: dict | None = None,
) -> AgentAnalyzeResponse:
    try:
        report = persist_analysis_result(
            db,
            run=run,
            response=response,
            questionnaire_responses=questionnaire_responses,
        )
    except Exception:
        logger.exception("Failed to persist analysis result for run %s", run.id)
        return response
    return response.model_copy(
        update={
            "analysis_id": run.id,
            "portfolio_id": run.portfolio_id,
            "report_id": report.id if report is not None else None,
        }
    )


@router.post("/agent/analyze", response_model=AgentAnalyzeResponse)
async def agent_analyze_endpoint(
    payload: AgentAnalyzeRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> AgentAnalyzeResponse:
    holdings = [h.model_dump() for h in payload.holdings]
    thread_id = str(uuid.uuid4())
    try:
        portfolio = ensure_analysis_portfolio(db, principal, payload)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GuestLimitExceeded as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        run = create_analysis_run(
            db, principal, portfolio=portfolio, thread_id=thread_id
        )
    except GuestLimitExceeded as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    pre_q = None
    if getattr(payload, "questionnaire_responses", None):
        try:
            pre_q = payload.questionnaire_responses.model_dump()
        except Exception:
            pre_q = payload.questionnaire_responses  # type: ignore
    initial_state = {
        "user_id": str(principal.id),
        "thread_id": thread_id,
        "confirmed_holdings": holdings,
        "history_period": payload.history_period,
        "history_interval": payload.history_interval,
        "analysis_status": "pending",
        "research_status": "pending",
        "critic_status": "pending",
        "behavioral_status": "pending",
        "human_review_status": "completed" if pre_q else "pending",
        "pending_questionnaire": False if pre_q else True,
        "final_advisor_status": "pending",
        "workflow_stage": "started",
        "tool_iterations": 0,
        "research_tool_iterations": 0,
        "max_tool_iterations": payload.max_tool_iterations,
        "warnings": [],
        "data_sources": [],
        "identified_risks": [],
        "tool_execution_summaries": [],
        "agent_messages": [],
        "research_messages": [],
        "market_data": {},
        "company_information": {},
        "retrieved_news": [],
        "research_sources": [],
        "research_warnings": [],
    }
    if pre_q:
        initial_state["behavioral_questionnaire"] = pre_q
    try:
        final_state = run_agent_analysis(initial_state, thread_id=thread_id)
    except Exception as exc:
        # Graph failed entirely; try to return deterministic metrics if possible
        provider_error = exc if isinstance(exc, LLMProviderError) else None
        logger.warning(
            "Agent workflow failed (provider=%s model=%s): %s",
            get_llm_info().provider,
            get_llm_info().model,
            exc,
        )
        try:
            metrics = analyze_portfolio(
                holdings=payload.holdings,
                history_period=payload.history_period,
                history_interval=payload.history_interval,
            )
            fallback = AgentAnalyzeResponse(
                portfolio_metrics=metrics,
                analyst_findings=None,
                research_findings=None,
                research_sources=[],
                research_status="failed",
                research_warnings=[f"Agent workflow failed before/during research: {exc}"],
                tool_execution_summaries=[],
                sources=[{"name": "yfinance", "type": "market_data", "detail": "Deterministic analytics fallback"}],
                warnings=[f"Agent workflow failed: {exc}"],
                analysis_status="partial",
                final_analysis=None,
                thread_id=thread_id,
                workflow_stage="failed",
                human_review_status="not_required",
                pending_questionnaire=False,
                critic_status="failed",
                behavioral_status="pending",
                final_advisor_status="failed",
            )
            return _persist_response(db, principal, fallback, run, pre_q)
        except Exception:
            if provider_error is not None:
                status, detail = _llm_http_status(provider_error)
                raise HTTPException(status_code=status, detail=detail) from exc
            raise HTTPException(status_code=502, detail="Agent analysis failed.")

    response = _build_agent_response(final_state, thread_id=thread_id)
    return _persist_response(db, principal, response, run, pre_q)


@router.post("/agent/resume", response_model=AgentAnalyzeResponse)
async def agent_resume_endpoint(
    payload: AgentResumeRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> AgentAnalyzeResponse:
    # Ownership is enforced from the persisted run before touching checkpoints.
    run = find_run_by_thread(db, payload.thread_id, principal)
    if run is None:
        raise HTTPException(status_code=404, detail="Invalid or unknown thread_id.")
    existing = get_analysis_state(payload.thread_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Invalid or unknown thread_id.")
    try:
        q = payload.questionnaire_responses
        q_dict = q.model_dump() if hasattr(q, "model_dump") else dict(q)
        # explicit validation
        BehavioralQuestionnaireResponse.model_validate(q_dict)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid questionnaire response: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid questionnaire response: {exc}") from exc

    try:
        final_state = resume_agent_analysis(payload.thread_id, q_dict)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        # try to return preserved state snapshot on resume failure
        state = get_analysis_state(payload.thread_id) or existing
        state = dict(state)
        state.setdefault("warnings", [])
        state["warnings"] = list(state.get("warnings") or []) + [f"Resume failed: {exc}"]
        state["analysis_status"] = state.get("analysis_status") or "partial"
        state["final_advisor_status"] = "failed"
        state["pending_questionnaire"] = False
        fallback = _build_agent_response(state, thread_id=payload.thread_id)
        return _persist_response(db, principal, fallback, run, q_dict)
    response = _build_agent_response(final_state, thread_id=payload.thread_id)
    return _persist_response(db, principal, response, run, q_dict)
