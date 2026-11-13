from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_optional_principal, require_user
from app.core.config import settings
from app.core.cookies import clear_guest_cookie, set_guest_cookie
from app.core.rate_limit import guest_rate_limit
from app.db.session import get_db
from app.models.auth import (
    AuthMeResponse,
    GuestConvertRequest,
    GuestConvertResponse,
    GuestCounts,
    GuestPublic,
)
from app.services.auth_service import (
    GuestConversionConfirmationRequired,
    GuestConversionConflict,
    Principal,
    convert_guest_to_user,
    create_guest_session,
    get_guest_principal,
    guest_data_counts,
)

router = APIRouter(prefix="/guest", tags=["guest"])


def _guest_public(guest) -> GuestPublic:
    return GuestPublic(
        id=guest.id,
        expires_at=guest.expires_at,
        created_at=guest.created_at,
    )


@router.post(
    "/session",
    response_model=AuthMeResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(guest_rate_limit)],
)
def create_guest(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthMeResponse:
    existing = get_guest_principal(db, request.cookies.get(settings.guest_cookie_name))
    if existing is not None and existing.guest is not None:
        counts = guest_data_counts(db, existing.id)
        return AuthMeResponse(
            authenticated=True,
            guest=_guest_public(existing.guest),
            guest_counts=GuestCounts(**counts),
            guest_session_expires_at=existing.guest.expires_at,
        )

    guest, token, expires_at = create_guest_session(
        db,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    set_guest_cookie(response, token, expires_at)
    return AuthMeResponse(
        authenticated=True,
        guest=_guest_public(guest),
        guest_counts=GuestCounts(),
        guest_session_expires_at=expires_at,
    )


@router.get("/me", response_model=AuthMeResponse)
def guest_me(
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal | None = Depends(get_optional_principal),
) -> AuthMeResponse:
    if principal is not None and principal.is_guest and principal.guest is not None:
        counts = guest_data_counts(db, principal.id)
        return AuthMeResponse(
            authenticated=True,
            guest=_guest_public(principal.guest),
            guest_counts=GuestCounts(**counts),
            guest_session_expires_at=principal.guest.expires_at,
        )
    return AuthMeResponse(authenticated=False)


@router.post("/convert", response_model=GuestConvertResponse)
def convert_guest(
    payload: GuestConvertRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_user),
) -> GuestConvertResponse:
    guest_principal = get_guest_principal(
        db, request.cookies.get(settings.guest_cookie_name)
    )
    if guest_principal is None or guest_principal.guest is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active guest session is available to convert.",
        )
    try:
        result = convert_guest_to_user(
            db,
            user=principal.user,
            guest=guest_principal.guest,
            confirm_merge=payload.confirm_merge,
        )
    except GuestConversionConfirmationRequired as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Confirm merging your guest work into this account.",
                "requires_confirmation": True,
                "counts": exc.counts,
            },
        ) from exc
    except GuestConversionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    clear_guest_cookie(response)
    counts = result.get("counts") or {}
    return GuestConvertResponse(
        converted=bool(result.get("converted", True)),
        already_converted=bool(result.get("already_converted", False)),
        counts=GuestCounts(**counts),
    )
