import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from 智能体.agent.refund_approval import (
    approval_thread_id, build_refund_approval_graph, resume_command,
)


def test_approval_pauses_then_resumes_with_program_generated_answer():
    graph = build_refund_approval_graph(InMemorySaver())
    config = {"configurable": {"thread_id": approval_thread_id(41)}}
    state = {
        "ticket_id": 41, "order_no": "DEMO-41", "eligible": "符合演示退款条件",
        "eligibility_note": "合成订单规则命中",
    }

    paused = graph.invoke(state, config)
    assert paused["__interrupt__"][0].value["ticket_id"] == 41

    completed = graph.invoke(resume_command("approve", "reviewer", "订单信息已核验"), config)
    assert "管理员已批准" in completed["answer"]
    assert "未发起真实退款" in completed["answer"]


def test_rejected_approval_is_not_reported_as_ineligible_order():
    graph = build_refund_approval_graph(InMemorySaver())
    config = {"configurable": {"thread_id": approval_thread_id(42)}}
    graph.invoke({
        "ticket_id": 42, "order_no": "DEMO-42", "eligible": "符合演示退款条件",
        "eligibility_note": "合成订单规则命中",
    }, config)
    completed = graph.invoke(Command(resume={"decision": "reject", "operator": "reviewer"}), config)
    assert completed["decision"] == "reject"
    assert "管理员未批准" in completed["answer"]
    assert "未发起真实退款" in completed["answer"]


def test_invalid_decision_is_rejected():
    try:
        resume_command("maybe", "reviewer")
    except ValueError as exc:
        assert "无效" in str(exc)
    else:
        raise AssertionError("invalid decisions must be rejected")


def test_postgres_checkpoint_can_resume_after_saver_recreation(store):
    thread_id = f"refund-restart-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "ticket_id": 999, "order_no": "DEMO-RESTART", "eligible": "符合演示退款条件",
        "eligibility_note": "合成订单规则命中",
    }
    with PostgresSaver.from_conn_string(store.dsn()) as saver:
        saver.setup()
        paused = build_refund_approval_graph(saver).invoke(initial, config)
        assert paused["__interrupt__"]

    # A new saver/connection simulates an application restart; the same thread resumes.
    with PostgresSaver.from_conn_string(store.dsn()) as saver:
        completed = build_refund_approval_graph(saver).invoke(
            resume_command("approve", "reviewer"), config
        )
    assert completed["decision"] == "approve"
    assert "未发起真实退款" in completed["answer"]


def test_refund_approval_rejects_duplicate_database_decision(clean_db):
    ticket = clean_db.sync_handoff_ticket("用户9921", "申请演示退款", {
        "need_human_reason": "模拟退款资格需要管理员审批",
        "urgency": "低", "emotion": "平静",
    })
    clean_db.set_ticket_status(ticket["id"], "待审批", operator="system")
    assert clean_db.decide_refund_approval(ticket["id"], "approve", "reviewer") is True
    assert clean_db.decide_refund_approval(ticket["id"], "reject", "reviewer") is False
