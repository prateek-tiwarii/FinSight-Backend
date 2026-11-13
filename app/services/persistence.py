from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.models import (
    AgentResult,
    AnalysisHoldingSnapshot,
    AnalysisRun,
    Holding,
    Portfolio,
    QuestionnaireResponse,
    Report,
    ResearchSource,
)
from app.models.agent import AgentAnalyzeResponse
from app.services.auth_service import Principal


class OwnershipError(Exception):
    pass


class NotFoundError(Exception):
    pass


class GuestLimitExceeded(Exception):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _owner_values(principal: Principal) -> dict[str, Any]:
    if principal.is_user:
        return {"user_id": principal.id, "guest_session_id": None}
    return {"user_id": None, "guest_session_id": principal.id}


def _owner_condition(principal: Principal):
    if principal.is_user:
        return Portfolio.user_id == principal.id
    return Portfolio.guest_session_id == principal.id


def _run_owner_condition(principal: Principal):
    if principal.is_user:
        return AnalysisRun.user_id == principal.id
    return AnalysisRun.guest_session_id == principal.id


def _report_owner_condition(principal: Principal):
    if principal.is_user:
        return Report.user_id == principal.id
    return Report.guest_session_id == principal.id


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep calculated values and closing-price context, but drop duplicated OHLCV."""
    if not metrics:
        return None
    compact = copy.deepcopy(metrics)
    for holding in compact.get("holdings", []) or []:
        hist = holding.get("historical_data")
        if isinstance(hist, dict):
            points = hist.get("points") or []
            hist["points"] = [
                {"date": p.get("date"), "close": p.get("close")}
                for p in points
                if isinstance(p, dict)
            ]
            hist["stored_note"] = "Only date/close retained for report snapshots."
    return compact


def analysis_dict(response: AgentAnalyzeResponse) -> dict[str, Any]:
    data = response.model_dump(mode="json")
    if data.get("portfolio_metrics"):
        data["portfolio_metrics"] = compact_metrics(data["portfolio_metrics"])
    return data


def get_portfolio_for_principal(
    db: Session, portfolio_id: uuid.UUID, principal: Principal
) -> Portfolio | None:
    return db.scalar(
        select(Portfolio)
        .options(selectinload(Portfolio.holdings))
        .where(Portfolio.id == portfolio_id, _owner_condition(principal))
    )


def replace_holdings(db: Session, portfolio: Portfolio, holdings: list[Any]) -> None:
    portfolio.holdings.clear()
    db.flush()
    for item in holdings:
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        portfolio.holdings.append(
            Holding(
                stock_symbol=str(data.get("stock_symbol", "")).strip(),
                quantity=_to_decimal(data.get("quantity")) or Decimal("0"),
                average_purchase_price=_to_decimal(data.get("average_purchase_price"))
                or Decimal("0"),
                exchange=data.get("exchange"),
            )
        )
    portfolio.updated_at = _utcnow()


def create_portfolio(
    db: Session,
    principal: Principal,
    *,
    name: str,
    holdings: list[Any],
) -> Portfolio:
    if principal.is_guest:
        count = int(
            db.scalar(
                select(func.count(Portfolio.id)).where(
                    Portfolio.guest_session_id == principal.id
                )
            )
            or 0
        )
        if count >= settings.guest_max_portfolios:
            raise GuestLimitExceeded(
                "Guest portfolio limit reached. Create an account to save more portfolios."
            )
    portfolio = Portfolio(name=name.strip() or "My Portfolio", **_owner_values(principal))
    db.add(portfolio)
    db.flush()
    replace_holdings(db, portfolio, holdings)
    db.commit()
    db.refresh(portfolio)
    return portfolio


def update_portfolio(
    db: Session,
    principal: Principal,
    portfolio: Portfolio,
    *,
    name: str | None,
    holdings: list[Any] | None,
) -> Portfolio:
    if name is not None:
        portfolio.name = name.strip() or portfolio.name
    if holdings is not None:
        replace_holdings(db, portfolio, holdings)
    portfolio.updated_at = _utcnow()
    db.commit()
    db.refresh(portfolio)
    return portfolio


def delete_portfolio(db: Session, principal: Principal, portfolio: Portfolio) -> None:
    db.delete(portfolio)
    db.commit()


def _latest_report_id(db: Session, portfolio_id: uuid.UUID) -> uuid.UUID | None:
    return db.scalar(
        select(Report.id)
        .join(AnalysisRun, Report.analysis_run_id == AnalysisRun.id)
        .where(AnalysisRun.portfolio_id == portfolio_id)
        .order_by(Report.created_at.desc())
        .limit(1)
    )


def portfolio_detail(db: Session, portfolio: Portfolio) -> dict[str, Any]:
    analysis_count = int(
        db.scalar(
            select(func.count(AnalysisRun.id)).where(
                AnalysisRun.portfolio_id == portfolio.id
            )
        )
        or 0
    )
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at,
        "holdings": [
            {
                "stock_symbol": h.stock_symbol,
                "quantity": float(h.quantity),
                "average_purchase_price": float(h.average_purchase_price),
                "exchange": h.exchange,
            }
            for h in portfolio.holdings
        ],
        "analysis_count": analysis_count,
        "latest_report_id": _latest_report_id(db, portfolio.id),
    }


def list_portfolios(db: Session, principal: Principal) -> list[dict[str, Any]]:
    portfolios = (
        db.scalars(
            select(Portfolio)
            .options(selectinload(Portfolio.holdings))
            .where(_owner_condition(principal))
            .order_by(Portfolio.updated_at.desc())
        )
        .unique()
        .all()
    )
    return [portfolio_detail(db, p) for p in portfolios]


def ensure_analysis_portfolio(
    db: Session, principal: Principal, payload: Any
) -> Portfolio:
    portfolio_id = getattr(payload, "portfolio_id", None)
    if portfolio_id:
        try:
            pid = uuid.UUID(str(portfolio_id))
        except ValueError as exc:
            raise NotFoundError("Portfolio not found.") from exc
        portfolio = get_portfolio_for_principal(db, pid, principal)
        if portfolio is None:
            raise NotFoundError("Portfolio not found.")
        replace_holdings(db, portfolio, payload.holdings)
        db.commit()
        db.refresh(portfolio)
        return portfolio

    name = getattr(payload, "portfolio_name", None) or f"Portfolio {_utcnow():%Y-%m-%d %H:%M}"
    return create_portfolio(db, principal, name=name, holdings=payload.holdings)


def create_analysis_run(
    db: Session,
    principal: Principal,
    *,
    portfolio: Portfolio,
    thread_id: str,
) -> AnalysisRun:
    if principal.is_guest:
        count = int(
            db.scalar(
                select(func.count(AnalysisRun.id)).where(
                    AnalysisRun.guest_session_id == principal.id
                )
            )
            or 0
        )
        if count >= settings.guest_max_analyses:
            raise GuestLimitExceeded(
                "Guest analysis limit reached. Create an account to run more analyses."
            )
    run = AnalysisRun(
        portfolio_id=portfolio.id,
        thread_id=thread_id,
        status="pending",
        workflow_stage="started",
        pending_questionnaire=False,
        **_owner_values(principal),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def find_run_by_id(db: Session, run_id: uuid.UUID, principal: Principal) -> AnalysisRun | None:
    return db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.portfolio))
        .where(AnalysisRun.id == run_id, _run_owner_condition(principal))
    )


def find_run_by_thread(
    db: Session, thread_id: str, principal: Principal
) -> AnalysisRun | None:
    return db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.portfolio))
        .where(
            AnalysisRun.thread_id == thread_id,
            _run_owner_condition(principal),
        )
    )


def _upsert_agent_result(
    db: Session,
    run: AnalysisRun,
    agent_name: str,
    status: str | None,
    findings: dict[str, Any] | None,
    sources: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> None:
    if findings is None and status is None:
        return
    existing = db.scalar(
        select(AgentResult).where(
            AgentResult.analysis_run_id == run.id,
            AgentResult.agent_name == agent_name,
        )
    )
    if existing is None:
        existing = AgentResult(analysis_run_id=run.id, agent_name=agent_name)
        db.add(existing)
    existing.status = status
    existing.findings_json = findings
    existing.sources_json = sources or []
    existing.warnings_json = warnings or []
    existing.updated_at = _utcnow()


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _persist_snapshots(db: Session, run: AnalysisRun, metrics: dict[str, Any] | None) -> None:
    db.query(AnalysisHoldingSnapshot).filter(
        AnalysisHoldingSnapshot.analysis_run_id == run.id
    ).delete(synchronize_session=False)
    if not metrics:
        return
    gen_at = _parse_timestamp(metrics.get("generated_at"))
    for h in metrics.get("holdings", []) or []:
        db.add(
            AnalysisHoldingSnapshot(
                analysis_run_id=run.id,
                input_symbol=h.get("input_symbol") or "",
                resolved_symbol=h.get("resolved_symbol"),
                quantity=_to_decimal(h.get("quantity")) or Decimal("0"),
                average_purchase_price=_to_decimal(h.get("average_purchase_price"))
                or Decimal("0"),
                market_price=_to_decimal(h.get("latest_price")),
                market_price_timestamp=gen_at,
                invested_value=_to_decimal(h.get("invested_value")),
                current_value=_to_decimal(h.get("current_value")),
                absolute_pnl=_to_decimal(h.get("absolute_pnl")),
                percentage_return=_to_decimal(h.get("percentage_return")),
                allocation_percent=_to_decimal(h.get("allocation_percent")),
                sector=h.get("sector"),
                status=h.get("status") or "unknown",
                message=h.get("message"),
                snapshot_json=copy.deepcopy(h),
            )
        )


def _persist_research_sources(
    db: Session, run: AnalysisRun, sources: list[dict[str, Any]], report: Report | None
) -> None:
    db.query(ResearchSource).filter(ResearchSource.analysis_run_id == run.id).delete(
        synchronize_session=False
    )
    seen: set[str] = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        key = str(source.get("url") or source.get("source_url") or source.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        db.add(
            ResearchSource(
                analysis_run_id=run.id,
                report_id=report.id if report else None,
                title=source.get("title") or source.get("headline"),
                url=source.get("url") or source.get("source_url"),
                publisher=source.get("publisher") or source.get("source_name"),
                publication_date=source.get("publication_date"),
                snippet=source.get("snippet") or source.get("summary"),
                metadata_json=copy.deepcopy(source),
            )
        )


def persist_analysis_result(
    db: Session,
    *,
    run: AnalysisRun,
    response: AgentAnalyzeResponse,
    questionnaire_responses: dict[str, Any] | None = None,
) -> Report | None:
    data = analysis_dict(response)
    run.status = response.analysis_status
    run.workflow_stage = response.workflow_stage
    run.pending_questionnaire = bool(response.pending_questionnaire)
    run.latest_result_json = data
    run.updated_at = _utcnow()
    if response.analysis_status in {"completed", "partial", "failed"}:
        run.completed_at = run.completed_at or _utcnow()

    metrics = data.get("portfolio_metrics")
    _persist_snapshots(db, run, metrics)

    _upsert_agent_result(
        db, run, "market_analyst", response.analysis_status,
        data.get("analyst_findings"), warnings=(data.get("analyst_findings") or {}).get("warnings"),
    )
    _upsert_agent_result(
        db, run, "research", response.research_status,
        data.get("research_findings"), sources=data.get("research_sources"),
        warnings=response.research_warnings,
    )
    _upsert_agent_result(
        db, run, "risk_critic", response.critic_status, data.get("critic_findings"),
    )
    _upsert_agent_result(
        db, run, "behavioral_advisor", response.behavioral_status,
        data.get("behavioral_findings"),
    )
    _upsert_agent_result(
        db, run, "final_advisor", response.final_advisor_status, data.get("final_report"),
    )

    if questionnaire_responses:
        row = db.scalar(
            select(QuestionnaireResponse).where(
                QuestionnaireResponse.analysis_run_id == run.id
            )
        )
        if row is None:
            row = QuestionnaireResponse(
                analysis_run_id=run.id, responses_json=questionnaire_responses
            )
            db.add(row)
        else:
            row.responses_json = questionnaire_responses
            row.updated_at = _utcnow()

    report = db.scalar(select(Report).where(Report.analysis_run_id == run.id))
    if response.final_report is not None:
        if report is None:
            owner = _owner_values(
                Principal(
                    kind="user" if run.user_id else "guest",
                    id=run.user_id or run.guest_session_id,
                )
            )
            report = Report(
                analysis_run_id=run.id,
                title=response.final_report.report_title,
                status=response.analysis_status,
                is_partial=response.analysis_status != "completed",
                report_json=data,
                metrics_json=metrics,
                **owner,
            )
            db.add(report)
            db.flush()
        # Reports are immutable: never overwrite report_json after creation.

    _persist_research_sources(
        db,
        run,
        list(data.get("research_sources") or []) + list(data.get("sources") or []),
        report,
    )
    db.commit()
    if report is not None:
        db.refresh(report)
    return report


def analysis_run_summary(db: Session, run: AnalysisRun) -> dict[str, Any]:
    report_id = db.scalar(select(Report.id).where(Report.analysis_run_id == run.id))
    return {
        "id": run.id,
        "portfolio_id": run.portfolio_id,
        "portfolio_name": run.portfolio.name if run.portfolio else "",
        "thread_id": run.thread_id,
        "status": run.status,
        "workflow_stage": run.workflow_stage,
        "pending_questionnaire": run.pending_questionnaire,
        "report_id": report_id,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def list_analysis_runs(db: Session, principal: Principal) -> list[dict[str, Any]]:
    runs = (
        db.scalars(
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.portfolio))
            .where(_run_owner_condition(principal))
            .order_by(AnalysisRun.created_at.desc())
        )
        .unique()
        .all()
    )
    return [analysis_run_summary(db, run) for run in runs]


def get_report_detail(db: Session, report: Report, principal: Principal) -> dict[str, Any]:
    run = db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.portfolio))
        .where(AnalysisRun.id == report.analysis_run_id)
    )
    if run is None:
        raise NotFoundError("Report not found.")
    sources = db.scalars(
        select(ResearchSource).where(ResearchSource.report_id == report.id)
    ).all()
    return {
        "id": report.id,
        "analysis_run_id": report.analysis_run_id,
        "portfolio_id": run.portfolio_id,
        "portfolio_name": run.portfolio.name if run.portfolio else "",
        "title": report.title,
        "status": report.status,
        "is_partial": report.is_partial,
        "created_at": report.created_at,
        "analysis": report.report_json,
        "metrics": report.metrics_json,
        "sources": [s.metadata_json or {} for s in sources],
    }


def list_reports(db: Session, principal: Principal) -> list[dict[str, Any]]:
    rows = db.execute(
        select(Report, AnalysisRun, Portfolio)
        .join(AnalysisRun, Report.analysis_run_id == AnalysisRun.id)
        .join(Portfolio, AnalysisRun.portfolio_id == Portfolio.id)
        .where(_report_owner_condition(principal))
        .order_by(Report.created_at.desc())
    ).all()
    return [
        {
            "id": report.id,
            "analysis_run_id": report.analysis_run_id,
            "portfolio_id": run.portfolio_id,
            "portfolio_name": portfolio.name,
            "title": report.title,
            "status": report.status,
            "is_partial": report.is_partial,
            "created_at": report.created_at,
        }
        for report, run, portfolio in rows
    ]


def get_report_for_principal(
    db: Session, report_id: uuid.UUID, principal: Principal
) -> Report | None:
    return db.scalar(
        select(Report).where(Report.id == report_id, _report_owner_condition(principal))
    )
