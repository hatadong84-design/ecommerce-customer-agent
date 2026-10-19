# -*- coding: utf-8 -*-
"""退款→转人工闭环的核心回归测试。

覆盖三件事：
1. 同一会话的重复诉求复用未关闭工单，不会无限开单；
2. 工单已解决后再提诉求会新开一张，而不是把已关的工单又打开；
3. 推动状态时已写的责任人 / 处理记录不会被清空。
"""
SESSION = "测试用户9001"


def _sentiment(**overrides):
    return {
        "need_human_reason": "用户要求退款，需人工核实订单",
        "urgency": "高",
        "emotion": "愤怒",
        **overrides,
    }


def test_sync_creates_ticket_then_reuses_it(clean_db):
    first = clean_db.sync_handoff_ticket(SESSION, "我要退款", _sentiment())
    assert first["created"] is True
    assert first["status"] == clean_db.TICKET_STATUSES[0]

    again = clean_db.sync_handoff_ticket(SESSION, "还是退款，补个订单号", _sentiment())
    assert again["created"] is False
    assert again["id"] == first["id"], "未关闭的工单应被复用，而不是重复开单"

    tickets = clean_db.query_tickets(session_id=SESSION)
    assert len(tickets) == 1
    assert "补个订单号" in tickets[0]["question"], "复用时应刷新为最新诉求"


def test_resolved_ticket_leads_to_new_one(clean_db):
    first = clean_db.sync_handoff_ticket(SESSION, "退款一次", _sentiment())
    clean_db.set_ticket_status(first["id"], "已解决", close_reason="已完成退款")

    second = clean_db.sync_handoff_ticket(SESSION, "退款二次", _sentiment())
    assert second["created"] is True
    assert second["id"] != first["id"], "已解决的工单不应被重新打开"
    assert len(clean_db.query_tickets(session_id=SESSION)) == 2


def test_status_update_keeps_existing_resolution(clean_db):
    ticket = clean_db.sync_handoff_ticket(SESSION, "退款", _sentiment())
    clean_db.set_ticket_status(
        ticket["id"], "处理中", agent_name="客服小李", resolution="已核对订单 #20260812"
    )
    # 关闭时只填关闭原因，之前写好的处理记录应保留（COALESCE 语义）
    clean_db.set_ticket_status(ticket["id"], "已解决", close_reason="用户认可，已到账")

    row = clean_db.query_tickets(session_id=SESSION)[0]
    assert row["status"] == "已解决"
    assert row["agent_name"] == "客服小李"
    assert row["resolution"] == "已核对订单 #20260812"
    assert row["close_reason"] == "用户认可，已到账"


def test_query_filters_and_counts(clean_db):
    in_progress = clean_db.sync_handoff_ticket("测试用户9002", "退款A", _sentiment())
    clean_db.sync_handoff_ticket("测试用户9003", "退款B", _sentiment())
    clean_db.set_ticket_status(in_progress["id"], "处理中")

    assert len(clean_db.query_tickets(session_id="测试用户9002")) == 1
    assert len(clean_db.query_tickets(status="处理中")) == 1
    assert clean_db.query_tickets(status="不存在的状态") == []

    counts = clean_db.ticket_counts()
    assert counts["待接入"] == 1
    assert counts["处理中"] == 1
    assert counts["已解决"] == 0


def test_session_ticket_map_points_to_latest(clean_db):
    first = clean_db.sync_handoff_ticket("测试用户9004", "退款", _sentiment())
    clean_db.set_ticket_status(first["id"], "已解决", close_reason="已处理")
    latest = clean_db.sync_handoff_ticket("测试用户9004", "退款再来", _sentiment())

    mapping = clean_db.session_ticket_map()
    assert mapping["测试用户9004"]["id"] == latest["id"]
    assert mapping["测试用户9004"]["status"] == clean_db.TICKET_STATUSES[0]
