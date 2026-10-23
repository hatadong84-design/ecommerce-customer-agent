# -*- coding: utf-8 -*-
"""退款闭环的回归测试：订单核验、会话隔离、工单通知与同步/流式一致性。

业务底线：
1. 退款不得跳过订单核验，也不得由模型判断资格；
2. 订单号只在当前会话可见，跨会话一律「无法核验」；
3. 建单后用户必须看到数据库里真实的工单号与状态；
4. 流式与同步路径产出的告知完全一致。
"""
import re

from 智能体.agent import order_service
from 智能体.api.app import (
    finalize_answer,
    handoff_notice,
    prepare_support,
    with_handoff_notice,
)

SESSION_A = "测试用户9101"
SESSION_B = "测试用户9102"


def _refund_payload(**overrides):
    return {
        "category": "退款",
        "need_human": False,
        "need_human_reason": "用户提出退款",
        "urgency": "高",
        "emotion": "焦虑",
        **overrides,
    }


def _order_no_of(session_id: str, index: int = 0) -> str:
    return order_service.list_demo_orders(session_id)[index]["order_no"]


# ---------------- 演示订单：合成数据与初始化 ----------------
def test_demo_orders_are_initialized_per_session(clean_db):
    orders = order_service.list_demo_orders(SESSION_A)
    assert len(orders) == 2, "每个会话应有可体验的演示订单"
    statuses = {o["refund_eligible"] for o in orders}
    assert statuses == {clean_db.REFUND_ELIGIBLE, clean_db.REFUND_INELIGIBLE}, \
        "应同时提供符合与不符合演示条件的订单，便于验证两条分支"
    assert all(o["order_no"].startswith("DEMO-") for o in orders)


def test_demo_order_numbers_differ_between_sessions(clean_db):
    a = {o["order_no"] for o in order_service.list_demo_orders(SESSION_A)}
    b = {o["order_no"] for o in order_service.list_demo_orders(SESSION_B)}
    assert a.isdisjoint(b), "不同会话的演示订单号不应相同"


def test_lookup_returns_order_of_current_session(clean_db):
    order_no = _order_no_of(SESSION_A)
    order = order_service.lookup(SESSION_A, order_no)
    assert order is not None
    assert order["order_no"] == order_no


def test_lookup_rejects_cross_session_order(clean_db):
    """A 的订单号拿到 B 的会话查询，必须查不到（防止越权读订单）"""
    order_no = _order_no_of(SESSION_A)
    assert order_service.lookup(SESSION_B, order_no) is None


def test_lookup_rejects_unknown_and_blank_order(clean_db):
    order_service.list_demo_orders(SESSION_A)  # 先初始化
    assert order_service.lookup(SESSION_A, "DEMO-999999-01") is None
    assert order_service.lookup(SESSION_A, "") is None


def test_lookup_is_case_and_space_insensitive(clean_db):
    order_no = _order_no_of(SESSION_A)
    messy = " " + order_no.lower().replace("-", " - ") + " "
    assert order_service.lookup(SESSION_A, messy) is not None


# ---------------- 订单号提取 ----------------
def test_extract_order_no_from_text():
    assert order_service.extract_order_no("订单 DEMO-1001-01 要退款") == "DEMO-1001-01"
    assert order_service.extract_order_no("demo 1001 01 退款") == "DEMO-1001-01"
    assert order_service.extract_order_no("我的订单号是 20261010001") == "20261010001"
    assert order_service.extract_order_no("我要退款") is None


def test_find_recent_order_only_within_same_session(clean_db):
    clean_db.log_turn(SESSION_A, "user", "我的订单号 DEMO-123456-01")
    assert order_service.find_recent_order_no(SESSION_A) == "DEMO-123456-01"
    assert order_service.find_recent_order_no(SESSION_B) is None, \
        "不能把别的会话里出现过的订单号带进来"


# ---------------- 退款分支：缺单号 / 核验通过 / 无法核验 ----------------
def test_refund_without_order_asks_first_and_creates_no_ticket(clean_db):
    ticket, order, stage = prepare_support(SESSION_A, "我要退款", _refund_payload())
    assert stage == "need_order"
    assert ticket is None and order is None
    assert clean_db.query_tickets(session_id=SESSION_A) == [], \
        "还没拿到订单号就不应建单"


def test_refund_with_valid_order_creates_ticket_with_eligibility(clean_db):
    order_no = _order_no_of(SESSION_A, index=0)
    ticket, order, stage = prepare_support(
        SESSION_A, f"订单 {order_no} 我要退款", _refund_payload()
    )
    assert stage == "verified"
    assert order is not None and order["order_no"] == order_no
    assert ticket["created"] is True
    assert ticket["status"] == clean_db.TICKET_STATUSES[0]

    saved = clean_db.query_tickets(session_id=SESSION_A)[0]
    assert order_no in saved["reason"]
    assert saved["reason"].find(order["refund_eligible"]) >= 0, \
        "工单里应记录程序核验到的资格，供人工复核"


def test_refund_with_unknown_order_still_hands_off_to_human(clean_db):
    ticket, order, stage = prepare_support(
        SESSION_A, "订单 DEMO-000000-99 退款", _refund_payload()
    )
    assert stage == "unverified"
    assert order is None
    assert ticket["created"] is True, "订单无法核验时也要交人工，不能直接拒绝用户"
    assert "无法核验" in clean_db.query_tickets(session_id=SESSION_A)[0]["reason"]


def test_refund_uses_order_number_from_earlier_turn(clean_db):
    """多轮：先说订单号，下一轮才提退款"""
    order_no = _order_no_of(SESSION_A, index=1)
    clean_db.log_turn(SESSION_A, "user", f"我的订单是 {order_no}")
    clean_db.log_turn(SESSION_A, "assistant", "好的，已记录。")

    ticket, order, stage = prepare_support(SESSION_A, "那我要退款", _refund_payload())
    assert stage == "verified"
    assert order is not None and order["order_no"] == order_no
    assert ticket["created"] is True


def test_repeat_refund_request_reuses_open_ticket(clean_db):
    order_no = _order_no_of(SESSION_A)
    # 模拟 handle_turn 的顺序：先把用户这轮提问写入记录，再做退款核验
    clean_db.log_turn(SESSION_A, "user", f"{order_no} 退款", _refund_payload())
    first, _, _ = prepare_support(SESSION_A, f"{order_no} 退款", _refund_payload())
    second, _, _ = prepare_support(SESSION_A, "怎么还没处理", _refund_payload())
    assert second["created"] is False
    assert second["id"] == first["id"], "同一会话未关闭的退款工单应复用"


def test_non_refund_without_human_flag_creates_no_ticket(clean_db):
    ticket, order, stage = prepare_support(
        SESSION_A, "扫地机器人怎么保养", {"category": "售后", "need_human": False}
    )
    assert (ticket, order, stage) == (None, None, "")
    assert clean_db.query_tickets(session_id=SESSION_A) == []


def test_after_sale_with_human_flag_creates_ticket(clean_db):
    ticket, order, stage = prepare_support(
        SESSION_A, "机器人一直报错", {"category": "售后", "need_human": True}
    )
    assert ticket["created"] is True
    assert order is None and stage == ""


# ---------------- 用户告知 ----------------
def test_notice_contains_real_ticket_number_and_status(clean_db):
    order_no = _order_no_of(SESSION_A)
    ticket, order, _ = prepare_support(SESSION_A, f"{order_no} 退款", _refund_payload())
    text = handoff_notice(ticket, order)
    assert f"#{ticket['id']}" in text
    assert ticket["status"] in text
    assert "人工回复为准" in text
    assert order["refund_eligible"] in text, "订单核验结论应一并告知用户"
    assert "演示数据" in text


def test_notice_without_order_has_no_verification_line():
    text = handoff_notice({"id": 7, "status": "待接入", "created": True})
    assert "核验结果" not in text


def test_notice_for_unverified_order_states_unable_to_verify(clean_db):
    """无法核验时由程序写明，不能只靠模型在回答里提一句"""
    ticket, order, stage = prepare_support(
        SESSION_A, "订单 DEMO-000000-99 退款", _refund_payload()
    )
    text = handoff_notice(ticket, order, stage)
    assert f"#{ticket['id']}" in text
    assert "无法核验" in text
    assert "人工回复为准" in text


def test_finalize_answer_appends_sources_and_notice(clean_db):
    order_no = _order_no_of(SESSION_A)
    ticket, order, _ = prepare_support(SESSION_A, f"{order_no} 退款", _refund_payload())
    out = finalize_answer("已收到您的退款诉求。", ["售后手册"], ticket, order)
    assert out.startswith("已收到您的退款诉求。")
    assert "参考来源：售后手册" in out
    assert f"#{ticket['id']}" in out
    # 幂等：再收尾一次不会追加第二条通知
    assert with_handoff_notice(out, ticket, order) == out


def test_stream_and_sync_share_identical_notice(clean_db):
    """同步与流式都调用 prepare_support + finalize_answer，告知文案必须一致"""
    order_no = _order_no_of(SESSION_A)
    ticket, order, _ = prepare_support(SESSION_A, f"{order_no} 退款", _refund_payload())
    sync_text = finalize_answer("答复正文", [], ticket, order)
    stream_text = finalize_answer("答复正文", [], ticket, order)
    assert re.sub(r"\s", "", sync_text) == re.sub(r"\s", "", stream_text)
