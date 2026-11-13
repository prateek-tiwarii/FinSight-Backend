from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.graph.checkpointer import (
    get_persistent_checkpointer,
    reset_persistent_checkpointer,
)
from app.graph.workflow import reset_agent_graph


def test_memory_checkpointer_is_used_in_tests():
    reset_persistent_checkpointer()
    reset_agent_graph()
    saver = get_persistent_checkpointer()
    assert isinstance(saver, MemorySaver)
    # Cached across calls so compiled graphs keep the same saver instance.
    assert get_persistent_checkpointer() is saver
    reset_persistent_checkpointer()
    reset_agent_graph()


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="Set TEST_POSTGRES_URL to run the PostgreSQL checkpointer test.",
)
def test_postgres_checkpointer_initializes(monkeypatch):
    reset_persistent_checkpointer()
    reset_agent_graph()
    monkeypatch.setattr(settings, "checkpoint_backend", "postgres")
    monkeypatch.setattr(settings, "checkpoint_dsn", os.environ["TEST_POSTGRES_URL"])
    saver = get_persistent_checkpointer()
    assert saver.__class__.__name__ == "PostgresSaver"
    reset_persistent_checkpointer()
    reset_agent_graph()
