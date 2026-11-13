from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from argon2.low_level import Type

_password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(email)))


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError, ValueError):
        return False
    except Exception:
        return False


def password_strength_errors(password: str) -> list[str]:
    errors: list[str] = []
    value = password or ""
    if len(value) < 10:
        errors.append("Password must be at least 10 characters long.")
    if not re.search(r"[a-z]", value):
        errors.append("Password must include a lowercase letter.")
    if not re.search(r"[A-Z]", value):
        errors.append("Password must include an uppercase letter.")
    if not re.search(r"\d", value):
        errors.append("Password must include a number.")
    if not re.search(r"[^A-Za-z0-9]", value):
        errors.append("Password must include a symbol.")
    return errors


def generate_session_token() -> str:
    """Cryptographically secure opaque token (returned to client, never stored raw)."""
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def is_expired(value: datetime, now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    return as_utc(value) <= current
