from __future__ import annotations

import os

# Configure a deterministic, isolated test environment before importing the app.
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["CHECKPOINT_BACKEND"] = "memory"
os.environ["COOKIE_SECURE"] = "false"
os.environ["AUTH_REQUIRED"] = "true"
os.environ.pop("DIRECT_URL", None)
os.environ.pop("CHECKPOINT_DSN", None)

import pytest  # noqa: E402

from app.core.rate_limit import reset_rate_limits  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db import models  # noqa: E402,F401  (register tables)
from app.db.session import get_engine  # noqa: E402

Base.metadata.create_all(bind=get_engine())


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    reset_rate_limits()
    yield
    reset_rate_limits()
