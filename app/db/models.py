from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    json_type,
    utcnow,
)


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    sessions: Mapped[list["AuthSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    portfolios: Mapped[list["Portfolio"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class AuthSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "auth_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400))

    user: Mapped[User] = relationship(back_populates="sessions")


class GuestSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "guest_sessions"

    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    converted_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    converted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400))

    portfolios: Mapped[list["Portfolio"]] = relationship(
        back_populates="guest_session", cascade="all, delete-orphan"
    )


class Portfolio(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "portfolios"
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL) <> (guest_session_id IS NOT NULL)",
            name="ck_portfolios_exactly_one_owner",
        ),
        Index("ix_portfolios_user_created", "user_id", "created_at"),
        Index("ix_portfolios_guest_created", "guest_session_id", "created_at"),
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    guest_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("guest_sessions.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="My Portfolio")

    user: Mapped[User | None] = relationship(back_populates="portfolios")
    guest_session: Mapped[GuestSession | None] = relationship(back_populates="portfolios")
    holdings: Mapped[list["Holding"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    analyses: Mapped[list["AnalysisRun"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )


class Holding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "holdings"
    __table_args__ = (Index("ix_holdings_portfolio_symbol", "portfolio_id", "stock_symbol"),)

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stock_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    average_purchase_price: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    exchange: Mapped[str | None] = mapped_column(String(16))

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")


class AnalysisRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL) <> (guest_session_id IS NOT NULL)",
            name="ck_analysis_runs_exactly_one_owner",
        ),
        Index("ix_analysis_runs_user_created", "user_id", "created_at"),
        Index("ix_analysis_runs_guest_created", "guest_session_id", "created_at"),
        Index("ix_analysis_runs_portfolio_created", "portfolio_id", "created_at"),
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    guest_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("guest_sessions.id", ondelete="CASCADE"), index=True
    )
    thread_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    workflow_stage: Mapped[str | None] = mapped_column(String(64))
    pending_questionnaire: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    latest_result_json: Mapped[dict[str, Any] | None] = mapped_column(json_type())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    portfolio: Mapped[Portfolio] = relationship(back_populates="analyses")
    snapshots: Mapped[list["AnalysisHoldingSnapshot"]] = relationship(
        back_populates="analysis_run", cascade="all, delete-orphan"
    )
    agent_results: Mapped[list["AgentResult"]] = relationship(
        back_populates="analysis_run", cascade="all, delete-orphan"
    )
    report: Mapped["Report | None"] = relationship(
        back_populates="analysis_run", cascade="all, delete-orphan", uselist=False
    )
    questionnaire: Mapped["QuestionnaireResponse | None"] = relationship(
        back_populates="analysis_run", cascade="all, delete-orphan", uselist=False
    )
    research_sources: Mapped[list["ResearchSource"]] = relationship(
        back_populates="analysis_run", cascade="all, delete-orphan"
    )


class AnalysisHoldingSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "analysis_holdings_snapshots"
    __table_args__ = (Index("ix_snapshots_run_symbol", "analysis_run_id", "input_symbol"),)

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    input_symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    resolved_symbol: Mapped[str | None] = mapped_column(String(128))
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    average_purchase_price: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    market_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    market_price_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    invested_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    current_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    absolute_pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    percentage_return: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    allocation_percent: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    sector: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    message: Mapped[str | None] = mapped_column(Text)
    snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(json_type())

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="snapshots")


class AgentResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "agent_results"
    __table_args__ = (
        UniqueConstraint("analysis_run_id", "agent_name", name="uq_agent_results_run_agent"),
    )

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str | None] = mapped_column(String(32))
    findings_json: Mapped[dict[str, Any] | None] = mapped_column(json_type())
    sources_json: Mapped[list[dict[str, Any]] | None] = mapped_column(json_type())
    warnings_json: Mapped[list[str] | None] = mapped_column(json_type())

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="agent_results")


class Report(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "reports"
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL) <> (guest_session_id IS NOT NULL)",
            name="ck_reports_exactly_one_owner",
        ),
        UniqueConstraint("analysis_run_id", name="uq_reports_analysis_run"),
        Index("ix_reports_user_created", "user_id", "created_at"),
        Index("ix_reports_guest_created", "guest_session_id", "created_at"),
    )

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    guest_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("guest_sessions.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    is_partial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(json_type())

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="report")
    research_sources: Mapped[list["ResearchSource"]] = relationship(
        back_populates="report"
    )


class ResearchSource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_sources"
    __table_args__ = (
        Index("ix_research_sources_run", "analysis_run_id"),
        Index("ix_research_sources_report", "report_id"),
    )

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    report_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reports.id", ondelete="SET NULL"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(500))
    url: Mapped[str | None] = mapped_column(String(1000))
    publisher: Mapped[str | None] = mapped_column(String(300))
    publication_date: Mapped[str | None] = mapped_column(String(64))
    snippet: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(json_type())

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="research_sources")
    report: Mapped[Report | None] = relationship(back_populates="research_sources")


class QuestionnaireResponse(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "questionnaire_responses"
    __table_args__ = (
        UniqueConstraint("analysis_run_id", name="uq_questionnaire_analysis_run"),
    )

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    responses_json: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="questionnaire")


__all__ = [
    "User",
    "AuthSession",
    "GuestSession",
    "Portfolio",
    "Holding",
    "AnalysisRun",
    "AnalysisHoldingSnapshot",
    "AgentResult",
    "Report",
    "ResearchSource",
    "QuestionnaireResponse",
]
