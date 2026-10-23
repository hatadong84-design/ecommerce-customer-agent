# -*- coding: utf-8 -*-
"""工单变更留痕的回归测试：谁、在什么时候、把工单从什么状态改成了什么。

事件表只追加不更新，且与主表更新在同一个事务里完成；
测试直接查 ticket_events，不依赖页面渲染。
"""
from 智能体.admin import store


def _payload():
    return {
        "need_human": True,
        "need_human_reason": "用户要求人工跟进",
        "urgency": "高",
        "emotion": "焦虑",
    }


def _new_ticket(session_id: str = "测试用户9301") -> dict:
    return store.sync_handoff_ticket(session_id, "我要投诉", _payload())


def test_new_ticket_writes_creation_event(clean_db):
    ticket = _new_ticket()
    events = clean_db.ticket_events(ticket["id"])
    assert len(events) == 1
    event = events[0]
    assert event["from_status"] is None
    assert event["to_status"] == "待接入"
    assert event["operator"] == "system"
    assert "转人工" in event["note"]


def test_status_change_records_operator_and_resolution(clean_db):
    ticket = _new_ticket()
    clean_db.set_ticket_status(
        ticket["id"], "处理中", agent_name="客服小李",
        resolution="已联系用户，核实为滤网堵塞", operator="admin",
    )
    events = clean_db.ticket_events(ticket["id"])
    assert len(events) == 2
    event = events[-1]
    assert (event["from_status"], event["to_status"]) == ("待接入", "处理中")
    assert event["operator"] == "admin"
    assert event["agent_name"] == "客服小李"
    assert event["note"] == "已联系用户，核实为滤网堵塞"


def test_close_keeps_resolution_and_records_reason(clean_db):
    ticket = _new_ticket()
    clean_db.set_ticket_status(ticket["id"], "处理中", resolution="已指导用户更换滤网",
                               operator="admin")
    clean_db.set_ticket_status(ticket["id"], "已解决", close_reason="用户确认问题解决",
                               operator="admin")
    events = clean_db.ticket_events(ticket["id"])
    assert len(events) == 3
    assert events[-1]["from_status"] == "处理中"
    assert events[-1]["to_status"] == "已解决"
    assert events[-1]["note"] == "用户确认问题解决"

    row = clean_db.query_tickets(session_id="测试用户9301")[0]
    assert row["status"] == "已解决"
    assert row["resolution"] == "已指导用户更换滤网", "关闭时不能清掉已有处理记录"
    assert row["close_reason"] == "用户确认问题解决"


def test_repeat_request_reuses_ticket_and_logs_reuse(clean_db):
    first = _new_ticket()
    second = clean_db.sync_handoff_ticket("测试用户9301", "怎么还没处理", _payload())
    assert second["id"] == first["id"] and second["created"] is False
    events = clean_db.ticket_events(first["id"])
    assert len(events) == 2, "复用只追加一条复用事件，不应再写一条建单事件"
    assert events[-1]["note"] == "重复诉求，复用未关闭工单"
    assert events[-1]["from_status"] == events[-1]["to_status"] == first["status"]


def test_note_only_change_still_logs(clean_db):
    ticket = _new_ticket()
    clean_db.set_ticket_status(ticket["id"], "待接入", resolution="补充：用户提供了订单号",
                               operator="admin")
    events = clean_db.ticket_events(ticket["id"])
    assert len(events) == 2
    assert events[-1]["from_status"] == events[-1]["to_status"] == "待接入"
    assert events[-1]["note"] == "补充：用户提供了订单号"


def test_identical_save_does_not_create_noise_event(clean_db):
    """后台表单会回填已有内容，原样再保存一次不应产生重复事件"""
    ticket = _new_ticket()
    clean_db.set_ticket_status(ticket["id"], "处理中", agent_name="客服小李",
                               resolution="已联系用户", operator="admin")
    before = len(clean_db.ticket_events(ticket["id"]))
    clean_db.set_ticket_status(ticket["id"], "处理中", agent_name="客服小李",
                               resolution="已联系用户", operator="admin")
    assert len(clean_db.ticket_events(ticket["id"])) == before


def test_ticket_event_map_groups_multiple_tickets(clean_db):
    first = _new_ticket("测试用户9302")
    second = _new_ticket("测试用户9303")
    clean_db.set_ticket_status(second["id"], "处理中", operator="admin")

    grouped = clean_db.ticket_event_map([first["id"], second["id"]])
    assert set(grouped) == {first["id"], second["id"]}
    assert len(grouped[first["id"]]) == 1
    assert len(grouped[second["id"]]) == 2
    assert clean_db.ticket_event_map([]) == {}


def test_unknown_ticket_is_ignored(clean_db):
    _new_ticket()
    clean_db.set_ticket_status(999999, "处理中", operator="admin")
    assert clean_db.ticket_events(999999) == []
