from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from 智能体.admin import store
from 智能体.agent.answer_guard import format_context
from 智能体.agent.tools.agent_tools import (
    rag_summarize,
    lookup_demo_order,
    get_user_id,
    get_user_location,
    set_user_city,
    get_weather,
    get_current_month,
    fetch_external_data,
    current_session_id,
    current_ticket,
    current_order,
    current_refund_stage,
    current_knowledge,
)
from 智能体.model.factory import chat_model
from 智能体.prompt_loader import load_system_prompts


def build_system_message(state) -> list:
    """在系统提示词里补上「当前会话身份」与「知识库商品分类」。

    把当前用户 ID / 城市直接写进提示词，模型查询使用记录时就不必先猜一个 ID，
    即便它并行发起工具调用也不会拿错用户；读取失败时退回原始提示词，不影响对话。"""
    prompt = load_system_prompts()

    session_id = current_session_id() or "未知"
    city = "未知"
    try:
        city = store.get_user_city(current_session_id()) or "未知"
    except Exception:
        pass
    prompt += (
        f"\n\n# 当前会话身份\n当前用户ID：{session_id}；所在城市：{city}。\n"
        "查询使用记录时，user_id 一律使用上面的当前用户ID，不要另行猜测或调用其它来源。\n"
        "需要天气时使用上面的城市；若城市为「未知」，先询问用户所在城市并用 set_user_city 记录后再查询。"
    )

    try:
        names = "、".join(c["name"] for c in store.list_kb_categories())
    except Exception:
        names = ""
    if names:
        prompt += (
            f"\n\n# 知识库商品分类\n知识库当前包含以下商品分类：{names}。"
            "知识库列出的每个商品分类都属于当前服务范围；"
            "不得以「只服务扫地机器人」为由拒绝其他分类的问题。"
            "当用户问题明确属于某个分类时，调用 rag_summarize 请传入对应的 category；不确定时留空。"
        )

    ticket = current_ticket()
    if ticket:
        prompt += (
            f"\n\n# 本轮人工工单（内部信息，不要原样复述给用户）\n"
            f"本轮工单号为 #{ticket['id']}，当前状态：{ticket.get('status') or '待接入'}。"
            "如需在回答中提及工单，请直接使用这个编号，绝对不要自行编造或虚构工单号；"
            "不要输出「本轮」「该用户」这类内部说明文字，系统会在回答末尾自动追加正式工单通知。"
        )

    # 退款分支：把程序核验到的事实交给提示词，并明确禁止模型自行下结论
    stage = current_refund_stage()
    order = current_order()
    if stage == "need_order":
        prompt += (
            "\n\n# 本轮退款处理（内部信息，不要原样复述）\n"
            "用户提出了退款/退货诉求，但尚未提供订单号。请先请用户提供订单号"
            "（可提示其查看问答页上的「演示订单」），本轮没有创建工单，"
            "不要提及任何工单号，也不要承诺退款结果。"
        )
    elif stage == "verified" and order:
        prompt += (
            f"\n\n# 本轮退款核验（内部信息，不要原样复述）\n"
            f"演示订单 {order['order_no']} 已由程序核验：{order['refund_eligible']}"
            f"（{order['eligibility_note']}）。已转人工处理。"
            "你可以转述「订单已核验」及演示数据中的资格结果，但绝不得承诺退款成功、"
            "到账时间或具体金额，处理结论一律以人工回复为准。"
        )
    elif stage == "unverified":
        prompt += (
            "\n\n# 本轮退款核验（内部信息，不要原样复述）\n"
            "用户提供的订单号无法核验（不存在、不属于当前用户或不是演示订单）。"
            "请如实告知无法核验，建议核对订单号；已转人工跟进。"
            "不得猜测订单信息，不得承诺退款结果。"
        )

    knowledge_context = format_context(current_knowledge())
    if knowledge_context:
        prompt += (
            "\n\n# 本轮知识库预检索（内部信息，不要原样复述）\n"
            f"{knowledge_context}"
        )

    return [SystemMessage(content=prompt)] + list(state.get("messages", []))


agent = create_react_agent(
    model=chat_model,
    tools=[
        rag_summarize,
        lookup_demo_order,
        get_user_id,
        get_user_location,
        set_user_city,
        get_weather,
        get_current_month,
        fetch_external_data,
    ],
    prompt=build_system_message,
)

if __name__ == "__main__":
    question = "扫地机器人无法回充怎么办？"
    print(f"【提问】{question}")

    seen_ids = set()
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        for message in chunk["messages"]:
            if message.id in seen_ids:
                continue
            seen_ids.add(message.id)
            if getattr(message, "tool_calls", None):
                print(f"【调用工具】{[tc['name'] for tc in message.tool_calls]}")
            elif message.content:
                print(f"【{type(message).__name__}】{message.content}")