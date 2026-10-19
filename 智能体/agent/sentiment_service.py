from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from 智能体.model.factory import chat_model
from 智能体.prompt_loader import load_sentiment_prompts


class SentimentResult(BaseModel):
    """客服场景下对用户输入的会话情感分析结果"""

    emotion: Literal["高兴", "平静", "焦虑", "愤怒", "伤心", "失望", "中性"] = Field(
        description="情绪倾向"
    )
    emotion_level: int = Field(description="情绪强度，0-100 的整数，数值越大情绪越强烈", ge=0, le=100)
    urgency: Literal["低", "中", "高"] = Field(description="紧急程度")
    need_human: bool = Field(description="是否需要转接人工客服")
    need_human_reason: str = Field(description="判断依据，一句话")
    satisfaction: Literal["满意", "一般", "不满意", "无法判断"] = Field(description="满意程度")
    summary: str = Field(description="对用户诉求的一句话概括")


class SentimentService:
    """分析用户输入的情绪、紧急程度、是否需要转接人工以及满意程度"""

    def __init__(self):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", load_sentiment_prompts()),
                ("human", "用户消息：\n{message}"),
            ])
            | chat_model.with_structured_output(SentimentResult)
        )

    def analyze(self, message: str) -> SentimentResult:
        return self.chain.invoke({"message": message})


if __name__ == "__main__":
    service = SentimentService()
    for text in [
        "扫地机器人突然充不进电了，灯一闪一闪的，今晚急着用，你们能不能直接派人来看看！",
        "谢谢，按你说的方法换了滤网，现在吸力恢复正常了，太棒了。",
        "想问一下，扫地机器人适合养猫的家庭吗？",
    ]:
        print(f"【输入】{text}")
        print(f"【分析】{service.analyze(text)}\n")
