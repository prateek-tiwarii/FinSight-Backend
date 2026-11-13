from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Response

from app.core.config import settings


def _cookie_kwargs() -> dict:
    return {
        "path": "/",
        "secure": settings.secure_cookies,
        "samesite": settings.cookie_samesite,
        "domain": settings.cookie_domain,
    }


def set_auth_cookie(response: Response, token: str, expires_at: datetime) -> None:
    max_age = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        max_age=max_age,
        **_cookie_kwargs(),
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        secure=settings.secure_cookies,
        samesite=settings.cookie_samesite,
        domain=settings.cookie_domain,
    )


def set_guest_cookie(response: Response, token: str, expires_at: datetime) -> None:
    max_age = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        key=settings.guest_cookie_name,
        value=token,
        httponly=True,
        max_age=max_age,
        **_cookie_kwargs(),
    )


def clear_guest_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.guest_cookie_name,
        path="/",
        secure=settings.secure_cookies,
        samesite=settings.cookie_samesite,
        domain=settings.cookie_domain,
    )
