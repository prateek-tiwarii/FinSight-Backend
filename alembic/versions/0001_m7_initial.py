"""Initial M7 persistence schema.

Revision ID: 0001_m7_initial
Revises:
Create Date: 2026-11-13
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_m7_initial"
down_revision = None
branch_labels = None
depends_on = None


def json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("full_name", sa.String(length=200), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("ip_address", sa.String(length=64)),
        sa.Column("user_agent", sa.String(length=400)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("token_hash", name="uq_auth_sessions_token_hash"),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_token_hash", "auth_sessions", ["token_hash"])

    op.create_table(
        "guest_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("converted_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("converted_at", sa.DateTime(timezone=True)),
        sa.Column("ip_address", sa.String(length=64)),
        sa.Column("user_agent", sa.String(length=400)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("token_hash", name="uq_guest_sessions_token_hash"),
    )
    op.create_index("ix_guest_sessions_token_hash", "guest_sessions", ["token_hash"])
    op.create_index("ix_guest_sessions_converted_user_id", "guest_sessions", ["converted_user_id"])

    op.create_table(
        "portfolios",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("guest_session_id", sa.Uuid(), sa.ForeignKey("guest_sessions.id", ondelete="CASCADE")),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(user_id IS NOT NULL) <> (guest_session_id IS NOT NULL)",
            name="ck_portfolios_exactly_one_owner",
        ),
    )
    op.create_index("ix_portfolios_user_id", "portfolios", ["user_id"])
    op.create_index("ix_portfolios_guest_session_id", "portfolios", ["guest_session_id"])
    op.create_index("ix_portfolios_user_created", "portfolios", ["user_id", "created_at"])
    op.create_index("ix_portfolios_guest_created", "portfolios", ["guest_session_id", "created_at"])

    op.create_table(
        "holdings",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("portfolio_id", sa.Uuid(), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stock_symbol", sa.String(length=64), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 6), nullable=False),
        sa.Column("average_purchase_price", sa.Numeric(20, 6), nullable=False),
        sa.Column("exchange", sa.String(length=16)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_holdings_portfolio_id", "holdings", ["portfolio_id"])
    op.create_index("ix_holdings_portfolio_symbol", "holdings", ["portfolio_id", "stock_symbol"])

    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("portfolio_id", sa.Uuid(), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("guest_session_id", sa.Uuid(), sa.ForeignKey("guest_sessions.id", ondelete="CASCADE")),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("workflow_stage", sa.String(length=64)),
        sa.Column("pending_questionnaire", sa.Boolean(), nullable=False),
        sa.Column("latest_result_json", json_type()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(user_id IS NOT NULL) <> (guest_session_id IS NOT NULL)",
            name="ck_analysis_runs_exactly_one_owner",
        ),
        sa.UniqueConstraint("thread_id", name="uq_analysis_runs_thread_id"),
    )
    op.create_index("ix_analysis_runs_portfolio_id", "analysis_runs", ["portfolio_id"])
    op.create_index("ix_analysis_runs_user_id", "analysis_runs", ["user_id"])
    op.create_index("ix_analysis_runs_guest_session_id", "analysis_runs", ["guest_session_id"])
    op.create_index("ix_analysis_runs_thread_id", "analysis_runs", ["thread_id"])
    op.create_index("ix_analysis_runs_user_created", "analysis_runs", ["user_id", "created_at"])
    op.create_index("ix_analysis_runs_guest_created", "analysis_runs", ["guest_session_id", "created_at"])
    op.create_index("ix_analysis_runs_portfolio_created", "analysis_runs", ["portfolio_id", "created_at"])

    op.create_table(
        "analysis_holdings_snapshots",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("analysis_run_id", sa.Uuid(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("input_symbol", sa.String(length=128), nullable=False),
        sa.Column("resolved_symbol", sa.String(length=128)),
        sa.Column("quantity", sa.Numeric(20, 6), nullable=False),
        sa.Column("average_purchase_price", sa.Numeric(20, 6), nullable=False),
        sa.Column("market_price", sa.Numeric(20, 6)),
        sa.Column("market_price_timestamp", sa.DateTime(timezone=True)),
        sa.Column("invested_value", sa.Numeric(20, 6)),
        sa.Column("current_value", sa.Numeric(20, 6)),
        sa.Column("absolute_pnl", sa.Numeric(20, 6)),
        sa.Column("percentage_return", sa.Numeric(20, 8)),
        sa.Column("allocation_percent", sa.Numeric(20, 8)),
        sa.Column("sector", sa.String(length=128)),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text()),
        sa.Column("snapshot_json", json_type()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_snapshots_run_id", "analysis_holdings_snapshots", ["analysis_run_id"])
    op.create_index("ix_snapshots_run_symbol", "analysis_holdings_snapshots", ["analysis_run_id", "input_symbol"])

    op.create_table(
        "agent_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("analysis_run_id", sa.Uuid(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_name", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32)),
        sa.Column("findings_json", json_type()),
        sa.Column("sources_json", json_type()),
        sa.Column("warnings_json", json_type()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("analysis_run_id", "agent_name", name="uq_agent_results_run_agent"),
    )
    op.create_index("ix_agent_results_run_id", "agent_results", ["analysis_run_id"])

    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("analysis_run_id", sa.Uuid(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("guest_session_id", sa.Uuid(), sa.ForeignKey("guest_sessions.id", ondelete="CASCADE")),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("is_partial", sa.Boolean(), nullable=False),
        sa.Column("report_json", json_type(), nullable=False),
        sa.Column("metrics_json", json_type()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(user_id IS NOT NULL) <> (guest_session_id IS NOT NULL)",
            name="ck_reports_exactly_one_owner",
        ),
        sa.UniqueConstraint("analysis_run_id", name="uq_reports_analysis_run"),
    )
    op.create_index("ix_reports_run_id", "reports", ["analysis_run_id"])
    op.create_index("ix_reports_user_id", "reports", ["user_id"])
    op.create_index("ix_reports_guest_session_id", "reports", ["guest_session_id"])
    op.create_index("ix_reports_user_created", "reports", ["user_id", "created_at"])
    op.create_index("ix_reports_guest_created", "reports", ["guest_session_id", "created_at"])

    op.create_table(
        "research_sources",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("analysis_run_id", sa.Uuid(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_id", sa.Uuid(), sa.ForeignKey("reports.id", ondelete="SET NULL")),
        sa.Column("title", sa.String(length=500)),
        sa.Column("url", sa.String(length=1000)),
        sa.Column("publisher", sa.String(length=300)),
        sa.Column("publication_date", sa.String(length=64)),
        sa.Column("snippet", sa.Text()),
        sa.Column("metadata_json", json_type()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_research_sources_run", "research_sources", ["analysis_run_id"])
    op.create_index("ix_research_sources_report", "research_sources", ["report_id"])

    op.create_table(
        "questionnaire_responses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("analysis_run_id", sa.Uuid(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("responses_json", json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("analysis_run_id", name="uq_questionnaire_analysis_run"),
    )
    op.create_index("ix_questionnaire_responses_run", "questionnaire_responses", ["analysis_run_id"])


def downgrade() -> None:
    op.drop_table("questionnaire_responses")
    op.drop_table("research_sources")
    op.drop_table("reports")
    op.drop_table("agent_results")
    op.drop_table("analysis_holdings_snapshots")
    op.drop_table("analysis_runs")
    op.drop_table("holdings")
    op.drop_table("portfolios")
    op.drop_table("guest_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("users")
