# -*- coding: utf-8 -*-
"""pytest 公共夹具：把测试指向独立的 agent_test 库，避免污染开发数据。

踩坑记录：config_handler 在导入时会执行 load_dotenv(override=True)，
它会用 .env 里的开发库连接串覆盖掉环境变量。所以**必须先把
config_handler 导入完**，再覆盖 DATABASE_URL；顺序反了测试就会连上
开发库（实测导致 clean_db 的 TRUNCATE 清掉了开发数据）。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 这一步会触发 load_dotenv(override=True)，必须放在覆盖 DATABASE_URL 之前
import 智能体.config_handler  # noqa: F401
from psycopg.conninfo import conninfo_to_dict, make_conninfo

BASE_DSN = os.environ["DATABASE_URL"]        # 此时已是 .env 注入的开发库连接串
TEST_DB = "agent_test"
_base = conninfo_to_dict(BASE_DSN)
TEST_DSN = make_conninfo(**{**_base, "dbname": TEST_DB})

# 覆盖为测试库；store.dsn() 是调用时才读环境变量，所以后续操作都落在测试库
os.environ["DATABASE_URL"] = TEST_DSN

import psycopg


def _ensure_test_db() -> None:
    """测试库不存在就创建（幂等）；连不上或没权限时由调用方转为跳过"""
    with psycopg.connect(make_conninfo(**{**_base, "dbname": _base["dbname"]}), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')


@pytest.fixture(scope="session")
def store():
    """建好表（幂等）的 store 模块；数据库不可用时跳过依赖它的用例"""
    try:
        _ensure_test_db()
    except Exception as e:  # 未启动 Docker / 无建库权限
        pytest.skip(f"PostgreSQL 不可用，跳过数据库相关用例：{e}")

    from 智能体.admin import store as store_module

    # 硬保险：连接串不是测试库就直接失败，绝不让测试打到开发库
    actual_db = conninfo_to_dict(store_module.dsn())["dbname"]
    if actual_db != TEST_DB:
        raise RuntimeError(f"测试必须使用 {TEST_DB}，当前却指向 {actual_db}，已中止以防污染开发数据")

    store_module._inited = False
    store_module.init_db()
    return store_module


@pytest.fixture()
def clean_db(store):
    """每个用例前清空业务数据，保证用例之间互不干扰（只在测试库执行）"""
    with store.connect() as conn:
        conn.execute(
            "TRUNCATE admin_user, chat_log, handoff_ticket, handoff_ticket_event, "
            "usage_report, user_profile, demo_order RESTART IDENTITY CASCADE"
        )
    return store
