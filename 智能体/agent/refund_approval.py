"""Human approval checkpoint for synthetic refund eligibility (never executes refunds)."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict, total=False):
    ticket_id: int
    order_no: str
    eligible: str
    eligibility_note: str
    decision: str
    operator: str
    note: str
    answer: str


def _approval_node(state: ApprovalState) -> dict:
    decision = interrupt({
        "ticket_id": state["ticket_id"],
        "order_no": state["order_no"],
        "eligible": state["eligible"],
        "eligibility_note": state["eligibility_note"],
    })
    decision = decision if isinstance(decision, dict) else {"decision": decision}
    approved = decision.get("decision") == "approve"
    answer = (
        f"管理员已批准演示订单 {state['order_no']} 的模拟退款资格（程序核验：{state['eligible']}）。"
        if approved else
        f"管理员未批准演示订单 {state['order_no']} 的模拟退款资格；如需进一步处理，请联系人工客服。"
    )
    answer += "本系统仅演示资格核验与人工审批，未发起真实退款。"
    return {"decision": "approve" if approved else "reject",
            "operator": decision.get("operator", "管理员"),
            "note": decision.get("note", ""), "answer": answer}


def build_refund_approval_graph(checkpointer):
    graph = StateGraph(ApprovalState)
    graph.add_node("approval", _approval_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=checkpointer)


def approval_thread_id(ticket_id: int) -> str:
    return f"refund-eligibility-{ticket_id}"


def resume_command(decision: str, operator: str, note: str = "") -> Command:
    if decision not in {"approve", "reject"}:
        raise ValueError("无效审批决定")
    return Command(resume={"decision": decision, "operator": operator, "note": note})
