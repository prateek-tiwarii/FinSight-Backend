from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_optional_principal
from app.core.config import settings
from app.core.cookies import clear_auth_cookie, set_auth_cookie
from app.core.rate_limit import login_rate_limit, register_rate_limit
from app.db.session import get_db
from app.models.auth import (
    AuthActionResponse,
    AuthMeResponse,
    GuestCounts,
    GuestPublic,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    UserPublic,
)
from app.services.auth_service import (
    DuplicateEmailError,
    InvalidCredentialsError,
    Principal,
    authenticate_user,
    create_auth_session,
    get_guest_principal,
    guest_data_counts,
    guest_has_data,
    register_user,
    revoke_auth_session,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_public(user) -> UserPublic:
    return UserPublic(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        created_at=user.created_at,
    )


def _guest_context(request: Request, db: Session) -> tuple[bool, GuestCounts | None]:
    token = request.cookies.get(settings.guest_cookie_name)
    guest_principal = get_guest_principal(db, token)
    if guest_principal is None or guest_principal.guest is None:
        return False, None
    counts = guest_data_counts(db, guest_principal.id)
    return guest_has_data(db, guest_principal.id), GuestCounts(**counts)


@router.get("/me", response_model=AuthMeResponse)
def auth_me(
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal | None = Depends(get_optional_principal),
) -> AuthMeResponse:
    conversion_available, counts = _guest_context(request, db)
    if principal is not None and principal.is_user and principal.user is not None:
        return AuthMeResponse(
            authenticated=True,
            user=_user_public(principal.user),
            guest_conversion_available=conversion_available,
            guest_counts=counts,
            auth_session_expires_at=principal.auth_session.expires_at
            if principal.auth_session
            else None,
        )
    if principal is not None and principal.is_guest and principal.guest is not None:
        guest_counts = guest_data_counts(db, principal.id)
        return AuthMeResponse(
            authenticated=True,
            guest=GuestPublic(
                id=principal.guest.id,
                expires_at=principal.guest.expires_at,
                created_at=principal.guest.created_at,
            ),
            guest_conversion_available=False,
            guest_counts=GuestCounts(**guest_counts),
            guest_session_expires_at=principal.guest.expires_at,
        )
    return AuthMeResponse(authenticated=False)


@router.post(
    "/register",
    response_model=AuthActionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(register_rate_limit)],
)
def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthActionResponse:
    try:
        user, token, expires_at = register_user(
            db,
            full_name=payload.full_name,
            email=str(payload.email),
            password=payload.password,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    except DuplicateEmailError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    set_auth_cookie(response, token, expires_at)
    conversion_available, counts = _guest_context(request, db)
    return AuthActionResponse(
        user=_user_public(user),
        guest_conversion_available=conversion_available,
        guest_counts=counts,
    )


@router.post(
    "/login",
    response_model=AuthActionResponse,
    dependencies=[Depends(login_rate_limit)],
)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthActionResponse:
    try:
        user = authenticate_user(db, email=str(payload.email), password=payload.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    token, expires_at = create_auth_session(
        db,
        user=user,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    set_auth_cookie(response, token, expires_at)
    conversion_available, counts = _guest_context(request, db)
    return AuthActionResponse(
        user=_user_public(user),
        guest_conversion_available=conversion_available,
        guest_counts=counts,
    )


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> MessageResponse:
    token = request.cookies.get(settings.session_cookie_name)
    revoke_auth_session(db, token)
    clear_auth_cookie(response)
    return MessageResponse(message="Signed out.")
