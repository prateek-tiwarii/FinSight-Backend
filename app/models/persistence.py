from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.agent import AgentAnalyzeRequest
from app.models.analysis import ConfirmedHolding


class PortfolioCreateRequest(BaseModel):
    name: str = Field(default="My Portfolio", min_length=1, max_length=200)
    holdings: list[ConfirmedHolding] = Field(default_factory=list)


class PortfolioUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    holdings: list[ConfirmedHolding] | None = None


class PortfolioDetail(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    updated_at: datetime
    holdings: list[ConfirmedHolding] = Field(default_factory=list)
    analysis_count: int = 0
    latest_report_id: uuid.UUID | None = None


class PortfolioListResponse(BaseModel):
    portfolios: list[PortfolioDetail] = Field(default_factory=list)


class AnalysisCreateRequest(AgentAnalyzeRequest):
    portfolio_id: uuid.UUID | None = None
    portfolio_name: str | None = Field(default=None, max_length=200)


class AnalysisRunSummary(BaseModel):
    id: uuid.UUID
    portfolio_id: uuid.UUID
    portfolio_name: str
    thread_id: str
    status: str
    workflow_stage: str | None = None
    pending_questionnaire: bool = False
    report_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime


class AnalysisListResponse(BaseModel):
    analyses: list[AnalysisRunSummary] = Field(default_factory=list)


class AnalysisDetail(AnalysisRunSummary):
    analysis: dict[str, Any] | None = None


class ReportSummary(BaseModel):
    id: uuid.UUID
    analysis_run_id: uuid.UUID
    portfolio_id: uuid.UUID
    portfolio_name: str
    title: str
    status: str
    is_partial: bool
    created_at: datetime


class ReportListResponse(BaseModel):
    reports: list[ReportSummary] = Field(default_factory=list)


class ReportDetail(ReportSummary):
    analysis: dict[str, Any]
    metrics: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
