# -*- coding: utf-8 -*-
"""演示订单适配器：只使用合成数据，按会话隔离，不连接真实电商平台。

三条硬约束（对应退款闭环的验收要求）：

1. 订单号只在当前会话内可见。查询强制带 session_id，跨会话一律返回「无法核验」，
   并且「订单不存在」与「订单属于别人」给出完全相同的提示，避免泄露订单是否存在。
2. 退款资格由本适配器依据演示数据给出，模型只能转述，不能自行判断或承诺结果。
3. 不执行任何退款、支付或物流动作；最终处理一律由人工确认。
"""
import re

from 智能体.admin import store

# 页面与工具返回里统一标注，提醒用户这不是真实订单
DEMO_DATA_NOTICE = "演示数据，非真实订单"

# 演示订单号（如 DEMO-1001-01）；也接受带空格的写法
DEMO_ORDER_PATTERN = re.compile(
    r"\bDEMO[- ]?(\d{3,}|[A-Z0-9]{4,})[- ]?(\d{1,2})\b", re.IGNORECASE
)

# 通用订单号兜底：字母 + 至少 6 位数字，或连续 10 位以上数字。
# 这类号码可以提交给适配器，但演示数据里通常查不到，会走「无法核验」分支。
GENERIC_ORDER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{2,}[-\s]?\d{6,}|\d{10,})(?![A-Za-z0-9])"
)


def normalize_order_no(order_no: str) -> str:
    """去掉空白并转大写，保证用户输入的写法差异不影响匹配"""
    return re.sub(r"\s+", "", (order_no or "")).upper()


def extract_order_no(text: str) -> str | None:
    """从一句话里取第一个订单号；没有则返回 None

    演示订单统一规范成 DEMO-编号-序号（去掉用户输入的空格差异），
    其它编号只去空白，交给适配器判断能否核验。
    """
    text = text or ""
    match = DEMO_ORDER_PATTERN.search(text)
    if match:
        return f"DEMO-{match.group(1).upper()}-{int(match.group(2)):02d}"
    match = GENERIC_ORDER_PATTERN.search(text)
    return normalize_order_no(match.group(0)) if match else None


def find_recent_order_no(session_id: str, limit: int = 12) -> str | None:
    """在当前会话最近几轮用户消息里找订单号。

    支持「先报订单号、下一轮再说退款」的多轮场景；只看本会话，
    不会把别的会话里出现过的订单号带进来。
    """
    if not session_id:
        return None
    for message in reversed(store.conversation_history(session_id, limit=limit)):
        if message.get("role") != "user":
            continue
        found = extract_order_no(message.get("content", ""))
        if found:
            return found
    return None


def list_demo_orders(session_id: str) -> list[dict]:
    """当前会话可见的演示订单（首次调用时初始化合成数据）"""
    return store.ensure_demo_orders(session_id)


def lookup(session_id: str, order_no: str) -> dict | None:
    """查询当前会话的演示订单；查不到或不属于本会话时返回 None"""
    return store.get_demo_order(session_id, order_no)


def describe(order: dict) -> str:
    """把订单渲染成给模型看的文本。

    资格结论来自数据库中的演示数据，措辞明确标注「演示」与「最终由人工确认」，
    避免模型把演示资格当成已生效的处理结果。
    """
    amount = f"{float(order.get('amount') or 0):.2f}"
    return (
        f"{DEMO_DATA_NOTICE}。订单号 {order['order_no']}，商品 {order['product']}，"
        f"金额 {amount} 元，下单日期 {order['purchased_at']}，状态 {order['order_status']}。"
        f"程序核验结果：{order['refund_eligible']}（{order['eligibility_note']}）。"
        "该结果仅描述演示数据，不构成退款承诺，最终处理由人工确认。"
    )
