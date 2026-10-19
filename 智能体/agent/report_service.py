from langgraph.prebuilt import create_react_agent

from 智能体.agent.tools.agent_tools import (
    rag_summarize,
    get_user_id,
    get_current_month,
    fetch_external_data,
)
from 智能体.model.factory import chat_model
from 智能体.prompt_loader import load_report_prompts


class ReportService:
    """使用 report_prompt.txt 作为系统提示词的报告智能体"""

    def __init__(self):
        self.agent = create_react_agent(
            model=chat_model,
            tools=[get_user_id, get_current_month, fetch_external_data, rag_summarize],
            prompt=load_report_prompts(),
        )

    def generate_report(self, user_id: str = None, month: str = None) -> str:
        if user_id and month:
            question = (
                f"请为{user_id}生成 {month} 的扫地机器人使用情况报告与保养建议。"
                f"本次已知用户ID为 {user_id}、月份为 {month}，"
                f"请直接把它们作为 fetch_external_data 的入参，不要调用 get_user_id / get_current_month。"
            )
        else:
            question = "请生成本月的扫地机器人使用情况报告与保养建议。"

        result = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content


if __name__ == "__main__":
    report_service = ReportService()
    print(report_service.generate_report("1003", "2026-09"))
