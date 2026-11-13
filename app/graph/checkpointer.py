from __future__ import annotations

import atexit
import logging
from contextlib import ExitStack
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings

logger = logging.getLogger(__name__)

_stack = ExitStack()
_checkpointer: Any | None = None


def is_postgres_url(url: str | None) -> bool:
    if not url:
        return False
    return url.startswith("postgres://") or url.startswith("postgresql")


def _psycopg_dsn(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


def get_persistent_checkpointer() -> Any:
    """Return the process-wide LangGraph checkpointer.

    * PostgreSQL (Neon) databases use the official ``PostgresSaver`` so checkpoints,
      interrupts, and questionnaire resumes survive backend restarts.
    * If no PostgreSQL URL is configured (unit tests/local-only), an in-memory saver is
      used outside production. Production refuses to silently downgrade.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    backend = (settings.checkpoint_backend or "auto").strip().lower()
    dsn = settings.checkpoint_dsn or settings.database_url

    use_postgres = backend == "postgres" or (backend == "auto" and is_postgres_url(dsn))
    if use_postgres and dsn:
        if backend == "postgres" and not is_postgres_url(dsn):
            raise RuntimeError(
                "CHECKPOINT_BACKEND=postgres requires a PostgreSQL DATABASE_URL/CHECKPOINT_DSN."
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            cm = PostgresSaver.from_conn_string(_psycopg_dsn(dsn))
            saver = _stack.enter_context(cm)
            saver.setup()
            _checkpointer = saver
            logger.info("LangGraph checkpointer initialized with PostgresSaver.")
            return _checkpointer
        except Exception:
            if settings.is_production:
                raise
            logger.exception(
                "Postgres checkpointer could not be initialized; falling back to MemorySaver "
                "outside production."
            )

    if settings.is_production:
        raise RuntimeError(
            "Persistent LangGraph checkpointing requires a PostgreSQL DATABASE_URL in production."
        )

    logger.warning(
        "Using in-memory LangGraph checkpointer (no PostgreSQL DATABASE_URL configured). "
        "Checkpoints will not survive a backend restart."
    )
    _checkpointer = MemorySaver()
    return _checkpointer


def reset_persistent_checkpointer() -> None:
    global _checkpointer, _stack
    _checkpointer = None
    try:
        _stack.close()
    except Exception:
        logger.exception("Failed to close checkpointer resources")
    _stack = ExitStack()


atexit.register(reset_persistent_checkpointer)
