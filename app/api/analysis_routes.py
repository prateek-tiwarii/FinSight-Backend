from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import require_principal
from app.db.session import get_db
from app.models.persistence import AnalysisDetail, AnalysisListResponse
from app.services.auth_service import Principal
from app.services.persistence import (
    analysis_run_summary,
    find_run_by_id,
    list_analysis_runs,
)

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.get("", response_model=AnalysisListResponse)
def list_analyses_endpoint(
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> AnalysisListResponse:
    return AnalysisListResponse(analyses=list_analysis_runs(db, principal))


@router.get("/{analysis_id}", response_model=AnalysisDetail)
def get_analysis_endpoint(
    analysis_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> AnalysisDetail:
    run = find_run_by_id(db, analysis_id, principal)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found."
        )
    return AnalysisDetail(
        **analysis_run_summary(db, run), analysis=run.latest_result_json
    )
