# -*- coding: utf-8 -*-
"""建表幂等性与工单表结构的回归测试。

对应需求「工单后台补充责任人、处理记录和关闭原因」——这几列是后加的，
用 ALTER 幂等补列，所以要保证重复执行 init_db 不会失败、列确实存在。
"""
from 智能体.admin.store import USER_ID_PREFIX


def test_init_db_is_idempotent(store):
    """重复执行建表不应报错（线上重启会反复走这条路）"""
    store._inited = False
    store.init_db()
    store._inited = False
    store.init_db()


def test_handoff_ticket_has_closure_columns(store):
    with store.connect() as conn:
        columns = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'handoff_ticket'"
            ).fetchall()
        }
    assert {"agent_name", "resolution", "close_reason", "status"} <= columns


def test_new_user_id_uses_prefix_and_number(store):
    user_id = store.new_user_id()
    assert user_id.startswith(USER_ID_PREFIX)
    assert user_id[len(USER_ID_PREFIX):].isdigit()


def test_kb_index_states_cover_failure(store):
    """索引失败必须是一个可识别的状态，否则无法做重试"""
    assert store.KB_INDEX_FAILED in store.KB_INDEX_STATES
    assert store.KB_INDEX_DONE in store.KB_INDEX_STATES
