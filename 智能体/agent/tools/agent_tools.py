import csv
import json
import urllib.parse
import urllib.request
from contextvars import ContextVar
from datetime import datetime

from langchain_core.tools import tool

from 智能体.admin import store
from 智能体.admin.store import USER_ID_PREFIX
from 智能体.agent import order_service
from 智能体.config_handler import agent_conf
from 智能体.logger_hander import logger
from 智能体.path_toll import get_abs_path
from 智能体.rag import rag_service
from 智能体.rag.rag_service import RagSummarizeService

rag = RagSummarizeService()

# ------------------------------------------------------------
# 当前会话身份
# Web 层在调用智能体之前，把「当前会话 ID」写进 ContextVar，工具执行时读回。
# 这样工具拿到的永远是当前会话的真实身份，而不是随机或写死的模拟值。
# langchain 的 run_in_executor 会 copy_context，所以流式（astream）下同样能取到。
# ------------------------------------------------------------
_current_session_id: ContextVar[str] = ContextVar("current_session_id", default="")


def set_current_session(session_id: str) -> None:
    """Web 层调用智能体前设置当前会话 ID"""
    _current_session_id.set(session_id or "")


def current_session_id() -> str:
    return _current_session_id.get()


# 本轮已创建/复用的人工工单（id、status、created）。Web 层在调用智能体前写入，
# 让提示词里带上真实工单号，避免模型自己编造一个编号后与系统追加的通知对不上。
_current_ticket: ContextVar[dict] = ContextVar("current_ticket", default={})


def set_current_ticket(ticket: dict | None) -> None:
    """Web 层创建工单后设置本轮工单信息；没有工单时传 None 清空，防止串到下一轮"""
    _current_ticket.set(ticket or {})


def current_ticket() -> dict:
    return _current_ticket.get()


# 本轮退款核验上下文：订单（演示数据）与所处阶段。
# 阶段取值：need_order（缺订单号）/ verified（已核验）/ unverified（无法核验）。
# Web 层在调用智能体前写入，提示词据此约束模型，不允许它自行判断退款资格。
_current_order: ContextVar[dict] = ContextVar("current_order", default={})
_current_refund_stage: ContextVar[str] = ContextVar("current_refund_stage", default="")


def set_current_order(order: dict | None, stage: str = "") -> None:
    """记录本轮退款核验结果；没有退款诉求时传 None 清空，防止串到下一轮"""
    _current_order.set(order or {})
    _current_refund_stage.set(stage or "")


def current_order() -> dict:
    return _current_order.get()


def current_refund_stage() -> str:
    return _current_refund_stage.get()


# 本轮程序侧预检索的资料。Web 层与评测脚本在调用智能体前写入，
# 提示词据此直接引用，不再依赖模型自己决定是否检索、传什么分类。
_current_knowledge: ContextVar[dict] = ContextVar("current_knowledge", default={})


def set_current_knowledge(knowledge: dict | None) -> None:
    """记录本轮预检索结果；没有预检索时传 None 清空，防止串到下一轮"""
    _current_knowledge.set(knowledge or {})


def current_knowledge() -> dict:
    return _current_knowledge.get()


# Open-Meteo（免费、免 key）：先地理编码拿经纬度，再查实时天气
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO 天气代码 -> 中文描述
WEATHER_CODES = {
    0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹",
}


def _http_get_json(url: str, params: dict, timeout: int = 8) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "smart-agent-cs/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@tool(description="从专业知识库检索参考资料；category 为商品分类（如「扫地机器人」），不确定时留空")
def rag_summarize(query: str, category: str = "") -> str:
    return rag.rag_summarize(query, category or "")


@tool(description="按订单号查询当前用户的演示订单（仅支持演示订单，如 DEMO-1001-01）；"
                  "查不到或不属于当前用户时统一返回「无法核验」")
def lookup_demo_order(order_no: str) -> str:
    """演示订单查询：只返回当前会话的合成订单，绝不出让其他会话的数据"""
    order = order_service.lookup(current_session_id(), order_no)
    if not order:
        return ("无法核验该订单号：订单不存在、不属于当前用户，或不是演示订单。"
                "请核对订单号后重试，或说明需要人工核实。")
    return order_service.describe(order)


# 检索结果里引用来源的标记，Web 层据此把来源随答案一起呈现给用户
RAG_SOURCE_PREFIX = rag_service.RAG_SOURCE_PREFIX


def extract_rag_sources(tool_output: str) -> list[str]:
    """从 rag_summarize 的返回值里取出引用的文档标题列表"""
    text = str(tool_output or "")
    index = text.find(RAG_SOURCE_PREFIX)
    if index < 0:
        return []
    tail = text[index + len(RAG_SOURCE_PREFIX):].splitlines()
    if not tail:
        return []
    return [name.strip() for name in tail[0].split("、") if name.strip()]


@tool(description="查询指定城市的实时天气，以消息字符串形式返回；接口不可用时返回「暂时无法查询」，不会编造数据")
def get_weather(city: str) -> str:
    city = (city or "").strip()
    if not city:
        return "未提供城市名称，无法查询天气。"

    try:
        geo = _http_get_json(
            GEOCODE_URL, {"name": city, "count": 1, "language": "zh", "format": "json"}
        )
        results = geo.get("results") or []
        if not results:
            return f"暂时无法查询「{city}」的天气：未找到该城市，请确认城市名称是否正确。"

        location = results[0]
        forecast = _http_get_json(
            FORECAST_URL,
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
        )
    except Exception as e:
        logger.error(f"[get_weather]查询 {city} 天气失败：{str(e)}")
        return f"暂时无法查询{city}的天气，请稍后再试。"

    current = forecast.get("current") or {}
    if not current:
        return f"暂时无法查询{city}的天气，请稍后再试。"

    description = WEATHER_CODES.get(current.get("weather_code"), "未知天气")
    name = location.get("name") or city
    return (
        f"{name}当前天气：{description}，"
        f"气温{current.get('temperature_2m')}℃，"
        f"相对湿度{current.get('relative_humidity_2m')}%，"
        f"风速{current.get('wind_speed_10m')}km/h。"
    )


@tool(description="获取当前用户所在城市的名称，以纯字符串形式返回；返回「未知」表示用户尚未登记城市")
def get_user_location() -> str:
    return store.get_user_city(current_session_id()) or "未知"


@tool(description="记录用户告知的所在城市，供后续查询天气使用；入参 city 为城市名称")
def set_user_city(city: str) -> str:
    city = (city or "").strip()
    session_id = current_session_id()
    if not city or not session_id:
        return "未能记录城市信息。"
    store.set_user_city(session_id, city)
    return f"已记录用户所在城市：{city}。"


@tool(description="获取当前用户的真实ID，以纯字符串形式返回，格式如 用户1001")
def get_user_id() -> str:
    return current_session_id() or "（未知用户）"


@tool(description="获取当前月份，以 YYYY-MM 格式的纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


@tool(description="检索指定用户指定月份的扫地机器人使用记录，入参 user_id 形如 用户1001，month 为 YYYY-MM 格式字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    data_path = get_abs_path(agent_conf["external_data_path"])
    # 使用记录里的「用户ID」列存的是纯数字（1001），而全站统一用「用户1001」，
    # 这里剥掉前缀再比对，所以两种写法都能查到。
    target = str(user_id or "").replace(USER_ID_PREFIX, "").strip()
    if not target or not str(month or "").strip():
        return "缺少用户 ID 或月份，无法查询使用记录。"

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            matched = [
                row for row in reader
                if row["用户ID"] == target and row["月份"] == str(month)
            ]
    except FileNotFoundError:
        return f"未找到使用记录数据文件：{data_path}"

    if not matched:
        return f"未查询到用户 {user_id} 在 {month} 的使用记录。"

    header = " | ".join(matched[0].keys())
    records = "\n".join(" | ".join(row.values()) for row in matched)
    return f"共 {len(matched)} 条使用记录：\n{header}\n{records}"
