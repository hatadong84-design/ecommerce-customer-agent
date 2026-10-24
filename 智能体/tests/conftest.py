import os
import sys
from pathlib import Path

import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load the project's .env before capturing the developer DSN, then isolate tests.
import 智能体.config_handler  # noqa: E402,F401

BASE_DSN = os.environ["DATABASE_URL"]
_base = conninfo_to_dict(BASE_DSN)
TEST_DB = "agent_test"
TEST_DSN = make_conninfo(**{**_base, "dbname": TEST_DB})
os.environ["DATABASE_URL"] = TEST_DSN

import psycopg  # noqa: E402


def _ensure_test_db():
    admin_db = _base.get("dbname") or "postgres"
    with psycopg.connect(make_conninfo(**{**_base, "dbname": admin_db}), autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')


@pytest.fixture(scope="session")
def store():
    try:
        _ensure_test_db()
        from 智能体.admin import store as store_module
        store_module._inited = False
        store_module.init_db()
    except Exception as exc:
        pytest.skip(f"PostgreSQL test database unavailable: {exc}")
    return store_module


@pytest.fixture
def clean_db(store):
    with store.connect() as conn:
        conn.execute(
            "TRUNCATE admin_user, chat_log, handoff_ticket, handoff_ticket_event, "
            "usage_report, user_profile, demo_order RESTART IDENTITY CASCADE"
        )
    return store
