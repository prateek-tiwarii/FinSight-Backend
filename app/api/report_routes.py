from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import require_principal
from app.db.session import get_db
from app.models.persistence import ReportDetail, ReportListResponse
from app.services.auth_service import Principal
from app.services.persistence import (
    NotFoundError,
    get_report_detail,
    get_report_for_principal,
    list_reports,
)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=ReportListResponse)
def list_reports_endpoint(
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> ReportListResponse:
    return ReportListResponse(reports=list_reports(db, principal))


@router.get("/{report_id}", response_model=ReportDetail)
def get_report_endpoint(
    report_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> ReportDetail:
    report = get_report_for_principal(db, report_id, principal)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Report not found."
        )
    try:
        detail = get_report_detail(db, report, principal)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return ReportDetail(**detail)
