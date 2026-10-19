from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from 智能体.admin import store
from 智能体.agent.tools.agent_tools import (
    rag_summarize,
    get_user_id,
    get_user_location,
    set_user_city,
    get_weather,
    get_current_month,
    fetch_external_data,
    current_session_id,
    current_ticket,
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
            "当用户问题明确属于某个分类时，调用 rag_summarize 请传入对应的 category；不确定时留空。"
        )

    ticket = current_ticket()
    if ticket:
        prompt += (
            f"\n\n# 本轮人工工单\n本轮已为该用户创建/更新人工工单 #{ticket['id']}"
            f"（当前状态：{ticket.get('status') or '待接入'}）。"
            "如需在回答中提及工单，请直接使用这个编号，绝对不要自行编造或虚构工单号。"
            "系统会在回答末尾追加正式工单通知，你只需说明「已提交人工、后续由人工跟进」。"
        )

    return [SystemMessage(content=prompt)] + list(state.get("messages", []))


agent = create_react_agent(
    model=chat_model,
    tools=[
        rag_summarize,
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