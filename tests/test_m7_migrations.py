from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect

BACKEND_DIR = Path(__file__).resolve().parents[1]

EXPECTED_TABLES = {
    "users",
    "auth_sessions",
    "guest_sessions",
    "portfolios",
    "holdings",
    "analysis_runs",
    "analysis_holdings_snapshots",
    "agent_results",
    "reports",
    "research_sources",
    "questionnaire_responses",
}


def _run_alembic(*args: str, database_url: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    env["DIRECT_URL"] = ""
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


def test_alembic_initial_migration_upgrades_and_downgrades_sqlite(tmp_path):
    db_path = (tmp_path / "migration.db").as_posix()
    url = f"sqlite+pysqlite:///{db_path}"

    upgrade = _run_alembic("upgrade", "head", database_url=url)
    assert upgrade.returncode == 0, upgrade.stderr

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert EXPECTED_TABLES.issubset(tables)
        assert "alembic_version" in tables
    finally:
        engine.dispose()

    downgrade = _run_alembic("downgrade", "base", database_url=url)
    assert downgrade.returncode == 0, downgrade.stderr

    engine = create_engine(url)
    try:
        remaining = set(inspect(engine).get_table_names())
        assert not (EXPECTED_TABLES & remaining)
    finally:
        engine.dispose()
