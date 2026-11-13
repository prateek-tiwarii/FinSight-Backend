from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies import require_principal
from app.db.session import get_db
from app.models.persistence import (
    PortfolioCreateRequest,
    PortfolioDetail,
    PortfolioListResponse,
    PortfolioUpdateRequest,
)
from app.services.auth_service import Principal
from app.services.persistence import (
    GuestLimitExceeded,
    create_portfolio,
    delete_portfolio,
    get_portfolio_for_principal,
    list_portfolios,
    portfolio_detail,
    update_portfolio,
)

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found.")


@router.get("", response_model=PortfolioListResponse)
def list_portfolios_endpoint(
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> PortfolioListResponse:
    return PortfolioListResponse(portfolios=list_portfolios(db, principal))


@router.post("", response_model=PortfolioDetail, status_code=status.HTTP_201_CREATED)
def create_portfolio_endpoint(
    payload: PortfolioCreateRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> PortfolioDetail:
    try:
        portfolio = create_portfolio(
            db, principal, name=payload.name, holdings=payload.holdings
        )
    except GuestLimitExceeded as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return PortfolioDetail(**portfolio_detail(db, portfolio))


@router.get("/{portfolio_id}", response_model=PortfolioDetail)
def get_portfolio_endpoint(
    portfolio_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> PortfolioDetail:
    portfolio = get_portfolio_for_principal(db, portfolio_id, principal)
    if portfolio is None:
        raise _not_found()
    return PortfolioDetail(**portfolio_detail(db, portfolio))


@router.patch("/{portfolio_id}", response_model=PortfolioDetail)
def update_portfolio_endpoint(
    portfolio_id: uuid.UUID,
    payload: PortfolioUpdateRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> PortfolioDetail:
    portfolio = get_portfolio_for_principal(db, portfolio_id, principal)
    if portfolio is None:
        raise _not_found()
    data = payload.model_dump(exclude_unset=True)
    portfolio = update_portfolio(
        db,
        principal,
        portfolio,
        name=data.get("name"),
        holdings=data.get("holdings"),
    )
    return PortfolioDetail(**portfolio_detail(db, portfolio))


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio_endpoint(
    portfolio_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_principal),
) -> Response:
    portfolio = get_portfolio_for_principal(db, portfolio_id, principal)
    if portfolio is None:
        raise _not_found()
    delete_portfolio(db, principal, portfolio)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
