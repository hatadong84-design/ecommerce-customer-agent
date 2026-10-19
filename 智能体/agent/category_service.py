from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from 智能体.model.factory import chat_model
from 智能体.prompt_loader import load_category_prompts

# 业务类型枚举：管理后台的筛选项与这里保持一致
CATEGORIES = ["售前", "售后", "退款"]


class CategoryResult(BaseModel):
    """电商客服场景下，用户问题的业务类型判定结果"""

    category: Literal["售前", "售后", "退款"] = Field(description="问题所属业务类型")
    category_reason: str = Field(description="判定依据，一句话")


class CategoryState(TypedDict):
    """分类图的状态：进来一个问题，出去一个类型与判定依据"""

    question: str
    category: str
    category_reason: str


_chain = (
    ChatPromptTemplate.from_messages([
        ("system", load_category_prompts()),
        ("human", "用户消息：\n{question}"),
    ])
    | chat_model.with_structured_output(CategoryResult)
)


def classify_node(state: CategoryState) -> CategoryState:
    """分类节点：用结构化输出把模型答案锁死为 售前 / 售后 / 退款 之一"""
    result = _chain.invoke({"question": state["question"]})
    return {"category": result.category, "category_reason": result.category_reason}


def route_by_category(state: CategoryState) -> Literal["售前", "售后", "退款"]:
    """扳道工：按分类节点填好的 category 决定走哪个分支"""
    return state["category"]


# 三个分支节点：目前只做归类打标，是"分流挂载点"。
# 🌟 将来若要按类别做不同处理（售前接商品库、售后接知识库、退款自动核订单），
#    改这三个节点即可，图的骨架不用动。
def pre_sale_node(state: CategoryState) -> CategoryState:
    return {}


def after_sale_node(state: CategoryState) -> CategoryState:
    return {}


def refund_node(state: CategoryState) -> CategoryState:
    return {}


def build_category_graph():
    """拼装分类图：START → 分类节点 →（条件边分流）→ 售前/售后/退款 → END"""
    builder = StateGraph(state_schema=CategoryState)
    builder.add_node("classify_node", classify_node)
    builder.add_node("售前", pre_sale_node)
    builder.add_node("售后", after_sale_node)
    builder.add_node("退款", refund_node)

    builder.add_edge(START, "classify_node")
    builder.add_conditional_edges("classify_node", route_by_category)
    builder.add_edge("售前", END)
    builder.add_edge("售后", END)
    builder.add_edge("退款", END)
    return builder.compile()


class CategoryService:
    """用 LangGraph 判断用户问题属于 售前 / 售后 / 退款 中的哪一类"""

    def __init__(self):
        self.graph = build_category_graph()

    def classify(self, message: str) -> CategoryResult:
        result = self.graph.invoke({"question": message})
        return CategoryResult(
            category=result["category"], category_reason=result["category_reason"]
        )


if __name__ == "__main__":
    service = CategoryService()
    for text in [
        "你们的扫地机器人支持自动集尘吗？和 X1 比哪个更适合养猫的家庭？",
        "机器清扫时一直提示边刷缠绕，回充还老是失败，怎么处理？",
        "上周买的，用了一次就坏了，我要退货退款。",
    ]:
        result = service.classify(text)
        print(f"【输入】{text}")
        print(f"【分类】{result.category}（{result.category_reason}）\n")
