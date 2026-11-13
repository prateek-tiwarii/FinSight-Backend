from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.services.auth_service import (
    Principal,
    get_authenticated_principal,
    get_guest_principal,
)


def get_optional_principal(
    request: Request, db: Session = Depends(get_db)
) -> Principal | None:
    auth_token = request.cookies.get(settings.session_cookie_name)
    if auth_token:
        principal = get_authenticated_principal(db, auth_token)
        if principal is not None:
            return principal
    guest_token = request.cookies.get(settings.guest_cookie_name)
    if guest_token:
        return get_guest_principal(db, guest_token)
    return None


def get_guest_from_cookie(
    request: Request, db: Session = Depends(get_db)
) -> Principal | None:
    guest_token = request.cookies.get(settings.guest_cookie_name)
    return get_guest_principal(db, guest_token) if guest_token else None


def require_principal(
    principal: Principal | None = Depends(get_optional_principal),
) -> Principal:
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in or continue as a guest to use this feature.",
        )
    return principal


def require_user(
    principal: Principal = Depends(require_principal),
) -> Principal:
    if not principal.is_user or principal.user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An authenticated account is required for this action.",
        )
    return principal
