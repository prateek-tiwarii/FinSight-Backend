from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.security import generate_csrf_token

SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _cookie_kwargs() -> dict:
    return {
        "path": "/",
        "secure": settings.secure_cookies,
        "samesite": settings.cookie_samesite,
        "domain": settings.cookie_domain,
    }


class CsrfProtectionMiddleware(BaseHTTPMiddleware):
    """Double-submit CSRF protection for cookie-authenticated API calls.

    A readable CSRF cookie is issued on the first response and must be echoed in the
    ``X-CSRF-Token`` header for unsafe requests. Session and guest cookies remain
    HttpOnly; only the non-secret CSRF token is readable by JavaScript.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        is_api = path.startswith("/api/")
        cookie_token = request.cookies.get(settings.csrf_cookie_name)

        if is_api and request.method not in SAFE_METHODS:
            header_token = request.headers.get("x-csrf-token")
            if (
                not cookie_token
                or not header_token
                or not secrets.compare_digest(cookie_token, header_token)
            ):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "CSRF validation failed. Refresh the page and try again."},
                )

        response = await call_next(request)

        if is_api and not cookie_token:
            response.set_cookie(
                key=settings.csrf_cookie_name,
                value=generate_csrf_token(),
                httponly=False,
                max_age=60 * 60 * 24 * 30,
                **_cookie_kwargs(),
            )
        return response


def verify_csrf(request: Request) -> None:
    """Reusable dependency for routes that must explicitly enforce CSRF."""
    if request.method in SAFE_METHODS:
        return
    cookie_token = request.cookies.get(settings.csrf_cookie_name)
    header_token = request.headers.get("x-csrf-token")
    if (
        not cookie_token
        or not header_token
        or not secrets.compare_digest(cookie_token, header_token)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed. Refresh the page and try again.",
        )
