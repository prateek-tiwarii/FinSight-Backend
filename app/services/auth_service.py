from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    as_utc,
    generate_session_token,
    hash_password,
    hash_token,
    is_expired,
    normalize_email,
    verify_password,
)
from app.db.models import (
    AnalysisRun,
    AuthSession,
    GuestSession,
    Portfolio,
    Report,
    User,
)


class AuthError(Exception):
    pass


class DuplicateEmailError(AuthError):
    pass


class InvalidCredentialsError(AuthError):
    pass


class GuestConversionError(Exception):
    pass


class GuestConversionConflict(GuestConversionError):
    pass


class GuestConversionConfirmationRequired(GuestConversionError):
    def __init__(self, counts: dict[str, int]) -> None:
        super().__init__("Guest data exists and requires explicit confirmation before merging.")
        self.counts = counts


@dataclass
class Principal:
    kind: str  # "user" | "guest"
    id: uuid.UUID
    user: User | None = None
    guest: GuestSession | None = None
    auth_session: AuthSession | None = None

    @property
    def is_user(self) -> bool:
        return self.kind == "user"

    @property
    def is_guest(self) -> bool:
        return self.kind == "guest"


def _expires(days: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=max(1, int(days)))


def register_user(
    db: Session,
    *,
    full_name: str,
    email: str,
    password: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> tuple[User, str, datetime]:
    normalized = normalize_email(email)
    existing = db.scalar(select(User.id).where(func.lower(User.email) == normalized))
    if existing is not None:
        raise DuplicateEmailError("An account with this email already exists.")
    user = User(
        full_name=full_name.strip(),
        email=normalized,
        password_hash=hash_password(password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise DuplicateEmailError("An account with this email already exists.") from exc
    db.refresh(user)
    token, expires_at = create_auth_session(
        db, user=user, ip_address=ip_address, user_agent=user_agent
    )
    return user, token, expires_at


def authenticate_user(db: Session, *, email: str, password: str) -> User:
    normalized = normalize_email(email)
    user = db.scalar(select(User).where(func.lower(User.email) == normalized))
    if user is None or not verify_password(password, user.password_hash):
        raise InvalidCredentialsError("Invalid email or password.")
    return user


def create_auth_session(
    db: Session,
    *,
    user: User,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> tuple[str, datetime]:
    token = generate_session_token()
    expires_at = _expires(settings.auth_session_days)
    session = AuthSession(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=(user_agent or "")[:400] or None,
    )
    db.add(session)
    db.commit()
    return token, expires_at


def get_authenticated_principal(db: Session, token: str | None) -> Principal | None:
    if not token:
        return None
    session = db.scalar(
        select(AuthSession).where(AuthSession.token_hash == hash_token(token))
    )
    if session is None or session.revoked_at is not None or is_expired(session.expires_at):
        return None
    user = db.get(User, session.user_id)
    if user is None:
        return None
    session.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    return Principal(kind="user", id=user.id, user=user, auth_session=session)


def revoke_auth_session(db: Session, token: str | None) -> None:
    if not token:
        return
    session = db.scalar(
        select(AuthSession).where(AuthSession.token_hash == hash_token(token))
    )
    if session is not None and session.revoked_at is None:
        session.revoked_at = datetime.now(timezone.utc)
        db.commit()


def create_guest_session(
    db: Session,
    *,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> tuple[GuestSession, str, datetime]:
    token = generate_session_token()
    expires_at = _expires(settings.guest_session_days)
    guest = GuestSession(
        token_hash=hash_token(token),
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=(user_agent or "")[:400] or None,
    )
    db.add(guest)
    db.commit()
    db.refresh(guest)
    return guest, token, expires_at


def get_guest_principal(db: Session, token: str | None) -> Principal | None:
    if not token:
        return None
    guest = db.scalar(
        select(GuestSession).where(GuestSession.token_hash == hash_token(token))
    )
    if (
        guest is None
        or guest.revoked_at is not None
        or guest.converted_at is not None
        or is_expired(guest.expires_at)
    ):
        return None
    guest.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    return Principal(kind="guest", id=guest.id, guest=guest)


def guest_data_counts(db: Session, guest_id: uuid.UUID) -> dict[str, int]:
    portfolios = int(
        db.scalar(
            select(func.count(Portfolio.id)).where(Portfolio.guest_session_id == guest_id)
        )
        or 0
    )
    analyses = int(
        db.scalar(
            select(func.count(AnalysisRun.id)).where(
                AnalysisRun.guest_session_id == guest_id
            )
        )
        or 0
    )
    reports = int(
        db.scalar(
            select(func.count(Report.id)).where(Report.guest_session_id == guest_id)
        )
        or 0
    )
    return {"portfolios": portfolios, "analyses": analyses, "reports": reports}


def guest_has_data(db: Session, guest_id: uuid.UUID) -> bool:
    counts = guest_data_counts(db, guest_id)
    return any(value > 0 for value in counts.values())


def convert_guest_to_user(
    db: Session,
    *,
    user: User,
    guest: GuestSession,
    confirm_merge: bool,
) -> dict[str, object]:
    """Transfer all guest-owned rows to an authenticated account atomically.

    Idempotent: repeating the call for the same user is a no-op. Never overwrites
    existing account data; rows are re-owned, not replaced.
    """
    if guest.converted_at is not None or guest.converted_user_id is not None:
        if guest.converted_user_id == user.id:
            return {"converted": True, "already_converted": True, "counts": guest_data_counts(db, guest.id)}
        raise GuestConversionConflict(
            "This guest session has already been linked to a different account."
        )

    counts = guest_data_counts(db, guest.id)
    has_data = any(value > 0 for value in counts.values())
    if has_data and not confirm_merge:
        raise GuestConversionConfirmationRequired(counts)

    now = datetime.now(timezone.utc)
    try:
        db.execute(
            update(Portfolio)
            .where(Portfolio.guest_session_id == guest.id)
            .values(user_id=user.id, guest_session_id=None, updated_at=now)
        )
        db.execute(
            update(AnalysisRun)
            .where(AnalysisRun.guest_session_id == guest.id)
            .values(user_id=user.id, guest_session_id=None, updated_at=now)
        )
        db.execute(
            update(Report)
            .where(Report.guest_session_id == guest.id)
            .values(user_id=user.id, guest_session_id=None, updated_at=now)
        )
        guest.converted_user_id = user.id
        guest.converted_at = now
        guest.revoked_at = now
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"converted": True, "already_converted": False, "counts": counts}


def session_is_expired(value: datetime) -> bool:
    return as_utc(value) <= datetime.now(timezone.utc)
