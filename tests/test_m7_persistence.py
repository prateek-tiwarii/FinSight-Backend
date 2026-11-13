from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from app.core.config import settings
from app.core.security import hash_token
from app.db.models import (
    AnalysisHoldingSnapshot,
    GuestSession,
    QuestionnaireResponse,
    Report,
)
from app.db.session import get_session_factory
from app.models.agent import AgentAnalyzeResponse, FinalPortfolioReport
from app.models.analysis import (
    HistoricalData,
    HoldingAnalysis,
    OHLCVPoint,
    PortfolioAnalyzeResponse,
    PortfolioSummary,
    RiskMetrics,
)
from app.services.auth_service import Principal
from app.services.persistence import (
    GuestLimitExceeded,
    create_analysis_run,
    create_portfolio,
    persist_analysis_result,
)


def _guest_principal(db) -> Principal:
    guest = GuestSession(
        token_hash=hash_token(uuid.uuid4().hex),
        expires_at=datetime.now(timezone.utc).replace(year=2099),
        last_seen_at=datetime.now(timezone.utc),
    )
    db.add(guest)
    db.commit()
    db.refresh(guest)
    return Principal(kind="guest", id=guest.id, guest=guest)


def _metrics(close: float = 110.0) -> PortfolioAnalyzeResponse:
    return PortfolioAnalyzeResponse(
        holdings=[
            HoldingAnalysis(
                input_symbol="AAA",
                resolved_symbol="AAA.NS",
                quantity=10,
                average_purchase_price=100,
                invested_value=1000,
                latest_price=close,
                current_value=1000 * close / 100,
                absolute_pnl=1000 * close / 100 - 1000,
                percentage_return=close - 100,
                allocation_percent=100,
                status="ok",
                historical_data=HistoricalData(
                    symbol="AAA",
                    period="1y",
                    interval="1d",
                    status="ok",
                    points=[
                        OHLCVPoint(
                            date="2026-01-01",
                            open=100,
                            high=115,
                            low=99,
                            close=close,
                            volume=12345,
                        )
                    ],
                ),
            )
        ],
        summary=PortfolioSummary(
            total_invested_amount=1000,
            total_current_value=1000 * close / 100,
            total_absolute_pnl=1000 * close / 100 - 1000,
            overall_percentage_return=close - 100,
            analyzed_holdings_count=1,
            unresolved_or_failed_holdings_count=0,
        ),
        risk=RiskMetrics(),
        generated_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


def _response(title: str, close: float = 110.0) -> AgentAnalyzeResponse:
    return AgentAnalyzeResponse(
        portfolio_metrics=_metrics(close),
        analysis_status="completed",
        workflow_stage="completed",
        pending_questionnaire=False,
        final_analysis={"summary": "Summary"},
        final_report=FinalPortfolioReport.model_construct(report_title=title),
        final_advisor_status="completed",
    )


def test_persist_result_creates_snapshot_and_immutable_report():
    db = get_session_factory()()
    principal = _guest_principal(db)
    portfolio = create_portfolio(
        db,
        principal,
        name="Persisted",
        holdings=[
            {
                "stock_symbol": "AAA",
                "quantity": 10,
                "average_purchase_price": 100,
            }
        ],
    )
    run = create_analysis_run(
        db, principal, portfolio=portfolio, thread_id=str(uuid.uuid4())
    )

    report = persist_analysis_result(
        db,
        run=run,
        response=_response("Original title"),
        questionnaire_responses={"risk_tolerance": "moderate"},
    )
    assert report is not None
    assert report.title == "Original title"

    snapshot = db.scalar(
        select(AnalysisHoldingSnapshot).where(
            AnalysisHoldingSnapshot.analysis_run_id == run.id
        )
    )
    assert snapshot is not None
    # SQLite drops tzinfo; compare the stored wall-clock value.
    assert snapshot.market_price_timestamp.replace(tzinfo=timezone.utc) == datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )
    assert snapshot.snapshot_json["historical_data"]["points"] == [
        {"date": "2026-01-01", "close": 110.0}
    ]
    assert (
        snapshot.snapshot_json["historical_data"]["stored_note"]
        == "Only date/close retained for report snapshots."
    )

    q = db.scalar(
        select(QuestionnaireResponse).where(
            QuestionnaireResponse.analysis_run_id == run.id
        )
    )
    assert q is not None
    assert q.responses_json == {"risk_tolerance": "moderate"}

    # A later resume/re-run must never rewrite the stored report snapshot.
    persist_analysis_result(db, run=run, response=_response("Changed title", close=120))
    db.refresh(report)
    reports = db.scalars(select(Report).where(Report.analysis_run_id == run.id)).all()
    assert len(reports) == 1
    assert reports[0].title == "Original title"
    assert reports[0].report_json["portfolio_metrics"]["holdings"][0]["latest_price"] == 110.0
    db.close()


def test_guest_analysis_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(settings, "guest_max_analyses", 1)
    db = get_session_factory()()
    principal = _guest_principal(db)
    portfolio = create_portfolio(db, principal, name="Limit", holdings=[])
    create_analysis_run(db, principal, portfolio=portfolio, thread_id=str(uuid.uuid4()))
    with pytest.raises(GuestLimitExceeded):
        create_analysis_run(
            db, principal, portfolio=portfolio, thread_id=str(uuid.uuid4())
        )
    db.close()


def test_snapshots_are_owned_by_run_and_cleared_on_replacement():
    db = get_session_factory()()
    principal = _guest_principal(db)
    portfolio = create_portfolio(db, principal, name="Snapshots", holdings=[])
    run = create_analysis_run(
        db, principal, portfolio=portfolio, thread_id=str(uuid.uuid4())
    )
    persist_analysis_result(db, run=run, response=_response("One"))
    persist_analysis_result(db, run=run, response=_response("Two"))
    count = db.scalar(
        select(func.count(AnalysisHoldingSnapshot.id)).where(
            AnalysisHoldingSnapshot.analysis_run_id == run.id
        )
    )
    assert count == 1
    db.close()
