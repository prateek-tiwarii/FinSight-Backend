from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.core.config import settings


class SlidingWindowLimiter:
    def __init__(self, window_seconds: int, limit: int) -> None:
        self.window_seconds = max(1, int(window_seconds))
        self.limit = max(1, int(limit))
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                return False, retry_after
            events.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


_limiters: dict[str, SlidingWindowLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter(scope: str) -> SlidingWindowLimiter:
    with _limiters_lock:
        if scope not in _limiters:
            windows = {
                "register": settings.rate_limit_register,
                "login": settings.rate_limit_login,
                "guest": settings.rate_limit_guest,
            }
            _limiters[scope] = SlidingWindowLimiter(
                settings.rate_limit_window_seconds, windows.get(scope, 10)
            )
        return _limiters[scope]


def enforce_rate_limit(scope: str, request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    key = f"{scope}:{client}"
    limiter = _limiter(scope)
    configured = {
        "register": settings.rate_limit_register,
        "login": settings.rate_limit_login,
        "guest": settings.rate_limit_guest,
    }.get(scope)
    if configured is not None:
        limiter.limit = max(1, int(configured))
    limiter.window_seconds = max(1, int(settings.rate_limit_window_seconds))
    allowed, retry_after = limiter.hit(key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )


def register_rate_limit(request: Request) -> None:
    enforce_rate_limit("register", request)


def login_rate_limit(request: Request) -> None:
    enforce_rate_limit("login", request)


def guest_rate_limit(request: Request) -> None:
    enforce_rate_limit("guest", request)


def reset_rate_limits() -> None:
    for limiter in list(_limiters.values()):
        limiter.reset()
