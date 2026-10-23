import csv
import hashlib
import io
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from 智能体.config_handler import db_conf

# 建表语句（幂等）：psycopg3 一次 execute 只能跑一条，所以拆成列表逐条执行
SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS admin_user (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_log (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        created_at TEXT NOT NULL,
        session_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        emotion TEXT,
        emotion_level INTEGER,
        urgency TEXT,
        need_human SMALLINT,
        need_human_reason TEXT,
        satisfaction TEXT,
        summary TEXT,
        category TEXT,
        tools_used TEXT
    )
    """,
    # chat_log 若已存在，CREATE TABLE 不会补列，用 ALTER 显式补上（幂等）
    "ALTER TABLE chat_log ADD COLUMN IF NOT EXISTS category TEXT",
    "CREATE INDEX IF NOT EXISTS idx_chat_log_session ON chat_log (session_id)",
    """
    CREATE TABLE IF NOT EXISTS handoff_ticket (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id TEXT,
        question TEXT NOT NULL,
        reason TEXT,
        urgency TEXT,
        emotion TEXT,
        status TEXT NOT NULL,
        agent_name TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ticket_session ON handoff_ticket (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_ticket_status ON handoff_ticket (status)",
    # 工单处理闭环：责任人（agent_name 已有）、处理记录、关闭原因，用 ALTER 幂等补列
    "ALTER TABLE handoff_ticket ADD COLUMN IF NOT EXISTS resolution TEXT",
    "ALTER TABLE handoff_ticket ADD COLUMN IF NOT EXISTS close_reason TEXT",
    # 工单变更留痕：每次建单、状态变更、补充处理记录都追加一条事件，
    # 只追加不更新，保证「谁在什么时候改了什么」可追溯。
    """
    CREATE TABLE IF NOT EXISTS handoff_ticket_event (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ticket_id BIGINT NOT NULL,
        session_id TEXT,
        from_status TEXT,
        to_status TEXT NOT NULL,
        operator TEXT,
        agent_name TEXT,
        note TEXT,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ticket_event_ticket ON handoff_ticket_event (ticket_id)",
    """
    CREATE TABLE IF NOT EXISTS kb_category (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kb_document (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        source_type TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kb_document_category ON kb_document (category)",
    # 索引状态：记录每篇文档是否成功写入向量库，失败时保留错误信息以便重试
    "ALTER TABLE kb_document ADD COLUMN IF NOT EXISTS index_status TEXT",
    "ALTER TABLE kb_document ADD COLUMN IF NOT EXISTS index_error TEXT",
    "ALTER TABLE kb_document ADD COLUMN IF NOT EXISTS indexed_at TEXT",
    """
    CREATE TABLE IF NOT EXISTS usage_report (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id TEXT NOT NULL,
        month TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (user_id, month)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_report_user ON usage_report (user_id)",
    # 用户资料：目前只存所在城市，供聊天时查询天气使用（没有记录就询问用户，不再随机）
    """
    CREATE TABLE IF NOT EXISTS user_profile (
        session_id TEXT PRIMARY KEY,
        city TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    # 访客编号序列：新访客依次拿到 用户1001、用户1002……
    "CREATE SEQUENCE IF NOT EXISTS user_id_seq START 1001",
    # 演示订单：合成数据，按会话绑定，只用于演示退款核验流程。
    # 同一订单号在不同会话下各自独立，查询时强制带 session_id，
    # 因此跨会话永远查不到别人的订单。
    """
    CREATE TABLE IF NOT EXISTS demo_order (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        order_no TEXT NOT NULL,
        session_id TEXT NOT NULL,
        product TEXT NOT NULL,
        amount NUMERIC(10, 2) NOT NULL,
        purchased_at TEXT NOT NULL,
        order_status TEXT NOT NULL,
        refund_eligible TEXT NOT NULL,
        eligibility_note TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (session_id, order_no)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_demo_order_session ON demo_order (session_id)",
]

# 知识库文档的来源类型（上传的文件格式，或后台手工录入）
KB_SOURCE_TYPES = ["txt", "pdf", "word", "手动"]

# 知识库文档的索引状态：从未索引、已写入向量库、写入失败（可重试）
KB_INDEX_PENDING = "未索引"
KB_INDEX_DONE = "已索引"
KB_INDEX_FAILED = "索引失败"
KB_INDEX_STATES = [KB_INDEX_PENDING, KB_INDEX_DONE, KB_INDEX_FAILED]

# 转人工工单的状态流转：待接入 -> 处理中 -> 已解决
TICKET_STATUSES = ["待接入", "处理中", "已解决"]

# 访客在界面上按「用户NNNN」展示，编号由数据库序列分配，从 1001 起
USER_ID_PREFIX = "用户"

# 演示订单的退款资格：由程序依据合成数据给出，不接受模型自行判断
REFUND_ELIGIBLE = "符合演示退款条件"
REFUND_INELIGIBLE = "不符合演示退款条件"

# chat_log 的字段顺序，插入与导出 CSV 都按它来
COLUMNS = [
    "created_at", "session_id", "role", "content",
    "emotion", "emotion_level", "urgency", "need_human",
    "need_human_reason", "satisfaction", "summary", "category", "tools_used",
]

# 管理后台表格/CSV 的中文表头，Web 模板与 Streamlit 版共用
DISPLAY_COLUMNS = {
    "created_at": "时间",
    "session_id": "用户ID",
    "role": "角色",
    "content": "内容",
    "emotion": "情绪",
    "emotion_level": "情绪强度",
    "urgency": "紧急程度",
    "need_human": "需转接人工",
    "need_human_reason": "转接理由",
    "satisfaction": "满意程度",
    "summary": "诉求摘要",
    "category": "业务类型",
    "tools_used": "调用工具",
}

_inited = False


def dsn() -> str:
    """优先用环境变量 DATABASE_URL 覆盖 db.yml 里的连接串"""
    return os.getenv("DATABASE_URL") or db_conf["dsn"]


def connect() -> psycopg.Connection:
    """每次操作新建短连接，避免跨线程复用同一个连接"""
    return psycopg.connect(dsn(), row_factory=dict_row)


def init_db() -> None:
    """建表（幂等）；同一个进程内只真正执行一次"""
    global _inited
    if _inited:
        return
    with connect() as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
    # 把历史遗留的随机会话 ID 统一改成「用户NNNN」
    migrate_session_ids()
    _inited = True


def _next_user_id(conn: psycopg.Connection) -> str:
    """用数据库序列取下一个编号，多请求并发也不会重号"""
    value = conn.execute("SELECT nextval('user_id_seq')").fetchone()["nextval"]
    return f"{USER_ID_PREFIX}{value}"


def new_user_id() -> str:
    """给新访客分配一个形如「用户1001」的编号"""
    with connect() as conn:
        return _next_user_id(conn)


def migrate_session_ids() -> int:
    """把历史遗留的随机会话 ID（如 b5b8890c）改名成「用户NNNN」。

    按每个会话最早出现的时间排序依次编号；chat_log 与 handoff_ticket 一起改，
    保证工单与对话仍然对得上；已经是「用户」开头的会话保持不变。
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT session_id FROM chat_log WHERE session_id IS NOT NULL "
            "GROUP BY session_id ORDER BY MIN(id)"
        ).fetchall()
        changed = 0
        for row in rows:
            old = row["session_id"]
            if old.startswith(USER_ID_PREFIX):
                continue
            new = _next_user_id(conn)
            conn.execute(
                "UPDATE chat_log SET session_id = %s WHERE session_id = %s", (new, old)
            )
            conn.execute(
                "UPDATE handoff_ticket SET session_id = %s WHERE session_id = %s", (new, old)
            )
            changed += 1
        return changed


def log_turn(session_id: str, role: str, content: str,
             sentiment: Mapping[str, Any] | None = None,
             tools: Sequence[str] | None = None) -> None:
    """写入一轮对话；sentiment 为 None 时（助手回答）只存内容与调用过的工具"""
    sentiment = sentiment or {}
    need_human = sentiment.get("need_human")
    values = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        session_id,
        role,
        content,
        sentiment.get("emotion"),
        sentiment.get("emotion_level"),
        sentiment.get("urgency"),
        None if need_human is None else int(need_human),
        sentiment.get("need_human_reason"),
        sentiment.get("satisfaction"),
        sentiment.get("summary"),
        sentiment.get("category"),
        "、".join(tools) if tools else None,
    )
    placeholders = ", ".join(["%s"] * len(COLUMNS))
    with connect() as conn:
        conn.execute(f"INSERT INTO chat_log ({', '.join(COLUMNS)}) VALUES ({placeholders})", values)


def query_turns(*, session_id: str | None = None, need_human: bool | None = None,
                urgency: str | None = None, category: str | None = None,
                keyword: str | None = None,
                limit: int = 500) -> list[dict[str, Any]]:
    """按会话、业务类型、是否需转接人工、紧急程度、关键词筛选记录"""
    sql = "SELECT * FROM chat_log WHERE 1 = 1"
    params: list[Any] = []
    if session_id:
        sql += " AND session_id = %s"
        params.append(session_id)
    if category:
        sql += " AND category = %s"
        params.append(category)
    if need_human is not None:
        sql += " AND need_human = %s"
        params.append(int(need_human))
    if urgency:
        sql += " AND urgency = %s"
        params.append(urgency)
    if keyword:
        # PostgreSQL 的 LIKE 区分大小写，用 ILIKE 才对得上 SQLite 的默认行为
        sql += " AND content ILIKE %s"
        params.append(f"%{keyword}%")
    sql += " ORDER BY id DESC LIMIT %s"
    params.append(limit)

    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def conversation_history(session_id: str, limit: int = 20) -> list[dict[str, str]]:
    """取某会话最近的 user/assistant 消息，按时间正序，用于拼多轮上下文"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_log "
            "WHERE session_id = %s AND role IN ('user', 'assistant') "
            "ORDER BY id DESC LIMIT %s",
            (session_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def _write_ticket_event(conn, ticket_id: int, session_id: str | None,
                        from_status: str | None, to_status: str,
                        operator: str | None = None, agent_name: str | None = None,
                        note: str | None = None, now: str | None = None) -> None:
    """在同一个连接/事务里追加一条工单事件，与主表更新一起提交或回滚"""
    conn.execute(
        "INSERT INTO handoff_ticket_event (ticket_id, session_id, from_status, to_status, "
        "operator, agent_name, note, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (ticket_id, session_id, from_status, to_status, operator, agent_name, note,
         now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )


def sync_handoff_ticket(session_id: str, question: str, sentiment: Mapping[str, Any]) -> dict[str, Any]:
    """命中转人工时：该会话已有未解决工单就刷新它，否则新建一张待接入工单。

    返回该会话当前的工单信息（id / status / created），供 Web 层明确告知用户
    「工单已提交、当前处于什么状态」，而不是让问答像普通聊天一样结束。
    建单与复用都会写一条事件，保证工单从出现的那一刻起就有记录。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reason = sentiment.get("need_human_reason")
    urgency = sentiment.get("urgency")
    emotion = sentiment.get("emotion")
    with connect() as conn:
        existing = conn.execute(
            "SELECT id, status FROM handoff_ticket "
            "WHERE session_id = %s AND status <> %s ORDER BY id DESC LIMIT 1",
            (session_id, TICKET_STATUSES[-1]),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE handoff_ticket SET question = %s, reason = %s, urgency = %s, "
                "emotion = %s, updated_at = %s WHERE id = %s",
                (question, reason, urgency, emotion, now, existing["id"]),
            )
            _write_ticket_event(
                conn, existing["id"], session_id, existing["status"], existing["status"],
                operator="system", note="重复诉求，复用未关闭工单", now=now,
            )
            return {"id": existing["id"], "status": existing["status"], "created": False}
        row = conn.execute(
            "INSERT INTO handoff_ticket (session_id, question, reason, urgency, emotion, "
            "status, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id, status",
            (session_id, question, reason, urgency, emotion, TICKET_STATUSES[0], now, now),
        ).fetchone()
        _write_ticket_event(
            conn, row["id"], session_id, None, row["status"],
            operator="system", note="用户诉求触发转人工", now=now,
        )
        return {"id": row["id"], "status": row["status"], "created": True}


def session_ticket_map() -> dict[str, dict[str, Any]]:
    """会话 ID -> 该会话最新一张工单（id / status）。

    问答页、管理后台据此把「已生成工单」的提示与跳转链接指到具体工单，
    而不是只给一个布尔值；没有工单的会话不会出现在这里。
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ON (session_id) session_id, id, status "
            "FROM handoff_ticket WHERE session_id IS NOT NULL "
            "ORDER BY session_id, id DESC"
        ).fetchall()
        return {row["session_id"]: {"id": row["id"], "status": row["status"]} for row in rows}


def query_tickets(status: str | None = None, session_id: str | None = None,
                  limit: int = 200) -> list[dict[str, Any]]:
    """按状态、会话筛选工单，默认列出全部（最新的在前）"""
    sql = "SELECT * FROM handoff_ticket WHERE 1 = 1"
    params: list[Any] = []
    if status:
        sql += " AND status = %s"
        params.append(status)
    if session_id:
        sql += " AND session_id = %s"
        params.append(session_id)
    sql += " ORDER BY updated_at DESC, id DESC LIMIT %s"
    params.append(limit)
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_ticket(ticket_id: int) -> dict[str, Any] | None:
    """按 id 取单张工单，用于工单详情页"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM handoff_ticket WHERE id = %s", (ticket_id,)
        ).fetchone()
        return dict(row) if row else None


def set_ticket_status(ticket_id: int, status: str, agent_name: str | None = None,
                      resolution: str | None = None,
                      close_reason: str | None = None,
                      operator: str | None = None) -> None:
    """推进工单状态，并可选记录责任人（坐席）、处理记录与关闭原因。

    agent_name / resolution / close_reason 传 None 时保留原值（COALESCE），
    避免只是改个状态就把已经写好的处理记录清空。
    主表更新与事件写入在同一个事务里完成，行锁保证 from_status 读到的是真实旧值。"""
    with connect() as conn:
        current = conn.execute(
            "SELECT status, session_id, agent_name, resolution, close_reason "
            "FROM handoff_ticket WHERE id = %s FOR UPDATE",
            (ticket_id,),
        ).fetchone()
        if not current:
            return

        # 表单会把已有内容回填，只有「值与库里不同」才算一次变更，
        # 避免后台重复点保存刷出内容相同的事件。
        new_agent = agent_name if agent_name != current["agent_name"] else None
        new_resolution = resolution if resolution != current["resolution"] else None
        new_close = close_reason if close_reason != current["close_reason"] else None
        if (current["status"] == status
                and not any([new_agent, new_resolution, new_close])):
            return

        conn.execute(
            "UPDATE handoff_ticket SET status = %s, agent_name = COALESCE(%s, agent_name), "
            "resolution = COALESCE(%s, resolution), close_reason = COALESCE(%s, close_reason), "
            "updated_at = %s WHERE id = %s",
            (status, new_agent, new_resolution, new_close,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticket_id),
        )
        _write_ticket_event(
            conn, ticket_id, current["session_id"], current["status"], status,
            operator=operator or "system",
            agent_name=new_agent or current["agent_name"],
            note=new_resolution or new_close
            or ("状态更新" if status != current["status"] else "更新责任人"),
        )


def ticket_events(ticket_id: int) -> list[dict[str, Any]]:
    """某张工单的全部变更记录，按发生顺序正序"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM handoff_ticket_event WHERE ticket_id = %s ORDER BY id",
            (ticket_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def ticket_event_map(ticket_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """一次查出多张工单的变更记录，供工单页批量展示"""
    if not ticket_ids:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM handoff_ticket_event WHERE ticket_id = ANY(%s::bigint[]) "
            "ORDER BY ticket_id, id",
            (list(ticket_ids),),
        ).fetchall()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["ticket_id"], []).append(dict(row))
    return grouped


def ticket_counts() -> dict[str, int]:
    """各状态工单数量，用于管理后台的工单概览"""
    counts = {state: 0 for state in TICKET_STATUSES}
    with connect() as conn:
        for row in conn.execute(
            "SELECT status, COUNT(*) AS c FROM handoff_ticket GROUP BY status"
        ).fetchall():
            counts[row["status"]] = row["c"]
    return counts


# ============================================================
# 知识库：按商品分类管理文档（支持 txt / pdf / word 上传与手工录入）
# ------------------------------------------------------------
# kb_document 是知识库的"数据源"，每个分类对应一份独立的数据管理器；
# 文档增删改后由 rag 层的 kb_service 同步维护向量库分片。
# ============================================================
def list_kb_categories() -> list[dict[str, Any]]:
    """商品分类列表，附带该分类下的文档数量"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.name, c.created_at, COUNT(d.id) AS doc_count "
            "FROM kb_category c LEFT JOIN kb_document d ON d.category = c.name "
            "GROUP BY c.name, c.created_at ORDER BY c.created_at, c.name"
        ).fetchall()
        return [dict(row) for row in rows]


def kb_category_exists(name: str) -> bool:
    with connect() as conn:
        return conn.execute(
            "SELECT 1 FROM kb_category WHERE name = %s", (name,)
        ).fetchone() is not None


def create_kb_category(name: str) -> None:
    """新建商品分类；重名会抛 UniqueViolation，由调用方捕获并提示"""
    with connect() as conn:
        conn.execute(
            "INSERT INTO kb_category (name, created_at) VALUES (%s, %s)",
            (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def delete_kb_category(name: str) -> list[int]:
    """删除分类及其全部文档，返回被删文档 id，供调用方同步清理向量库"""
    with connect() as conn:
        ids = [row["id"] for row in conn.execute(
            "SELECT id FROM kb_document WHERE category = %s", (name,)
        ).fetchall()]
        conn.execute("DELETE FROM kb_document WHERE category = %s", (name,))
        conn.execute("DELETE FROM kb_category WHERE name = %s", (name,))
        return ids


def list_kb_documents(category: str | None = None,
                      with_content: bool = False) -> list[dict[str, Any]]:
    """分类下的文档清单；默认不带正文（列表页用），数据管理器页需要正文时传 with_content=True"""
    fields = ("id, category, title, source_type, length(content) AS char_count, "
              "created_at, updated_at, index_status, index_error")
    if with_content:
        fields += ", content"
    sql = f"SELECT {fields} FROM kb_document WHERE 1 = 1"
    params: list[Any] = []
    if category:
        sql += " AND category = %s"
        params.append(category)
    sql += " ORDER BY updated_at DESC, id DESC"
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_kb_document(doc_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM kb_document WHERE id = %s", (doc_id,)).fetchone()
        return dict(row) if row else None


def create_kb_document(category: str, title: str, source_type: str, content: str) -> int:
    """新增一篇知识文档，返回其 id（后续用它给向量库分片打标记）。
    新文档的索引状态先记为「未索引」，写完向量库后由 kb_service 改成「已索引」。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connect() as conn:
        row = conn.execute(
            "INSERT INTO kb_document (category, title, source_type, content, "
            "created_at, updated_at, index_status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (category, title, source_type, content, now, now, KB_INDEX_PENDING),
        ).fetchone()
        return row["id"]


def set_document_index_status(doc_id: int, status: str, error: str | None = None) -> None:
    """记录一篇文档的索引结果：成功记「已索引」，失败记「索引失败」并保留错误信息"""
    with connect() as conn:
        conn.execute(
            "UPDATE kb_document SET index_status = %s, index_error = %s, indexed_at = %s "
            "WHERE id = %s",
            (status, error, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), doc_id),
        )


def update_kb_document(doc_id: int, title: str, content: str) -> dict[str, Any] | None:
    """修改标题与正文；返回更新后的整行，供调用方重新索引"""
    with connect() as conn:
        conn.execute(
            "UPDATE kb_document SET title = %s, content = %s, updated_at = %s WHERE id = %s",
            (title, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), doc_id),
        )
        row = conn.execute("SELECT * FROM kb_document WHERE id = %s", (doc_id,)).fetchone()
        return dict(row) if row else None


def delete_kb_document(doc_id: int) -> dict[str, Any] | None:
    """删除文档并返回其原信息；返回 None 表示 id 不存在"""
    with connect() as conn:
        row = conn.execute("SELECT * FROM kb_document WHERE id = %s", (doc_id,)).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM kb_document WHERE id = %s", (doc_id,))
        return dict(row)


def kb_document_total() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM kb_document").fetchone()["c"]


# ============================================================
# 使用报告：管理员生成后落库，用户可在自己的问答页看到
# ------------------------------------------------------------
# 同一个「用户 + 月份」只保留一份报告，重名生成时覆盖（upsert），
# 所以重复点击不会堆积，用户页取的也永远是最新那份。
# ============================================================
def save_usage_report(user_id: str, month: str, content: str) -> None:
    """保存使用报告；同一用户同一月份重复生成时覆盖旧报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connect() as conn:
        conn.execute(
            "INSERT INTO usage_report (user_id, month, content, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (user_id, month) DO UPDATE SET "
            "content = EXCLUDED.content, updated_at = EXCLUDED.updated_at",
            (user_id, month, content, now, now),
        )


def list_usage_reports(user_id: str) -> list[dict[str, Any]]:
    """某个用户的报告列表（按月份倒序）"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, user_id, month, content, created_at, updated_at "
            "FROM usage_report WHERE user_id = %s ORDER BY month DESC",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


# ============================================================
# 用户资料：目前只存所在城市，供聊天时查询天气
# ------------------------------------------------------------
# 没有记录时返回 None，工具据此提示「未知」，由模型询问用户，
# 不再用随机城市或固定城市顶替。
# ============================================================
def get_user_city(session_id: str) -> str | None:
    if not session_id:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT city FROM user_profile WHERE session_id = %s", (session_id,)
        ).fetchone()
        return (row["city"] if row else None) or None


def set_user_city(session_id: str, city: str) -> None:
    """记录（或更新）某个会话所在的城市"""
    if not session_id:
        return
    with connect() as conn:
        conn.execute(
            "INSERT INTO user_profile (session_id, city, updated_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (session_id) DO UPDATE SET "
            "city = EXCLUDED.city, updated_at = EXCLUDED.updated_at",
            (session_id, city, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


# ============================================================
# 演示订单：合成数据，按会话绑定，仅用于演示退款核验流程
# ------------------------------------------------------------
# 不连接任何真实电商平台，也不执行退款；查询一律带 session_id，
# 保证某个会话拿不到另一个会话的订单。
# ============================================================
def demo_order_no(session_id: str, seq: int) -> str:
    """生成演示订单号，形如 DEMO-1001-01。

    编号带上会话的数字部分，便于手工输入；会话没有数字时用短哈希占位。
    同一个编号在不同会话下也是相互独立的记录（唯一约束是会话 + 订单号）。
    """
    digits = "".join(ch for ch in (session_id or "") if ch.isdigit())
    if digits:
        code = digits[-6:]
    else:
        code = hashlib.sha1((session_id or "demo").encode("utf-8")).hexdigest()[:6].upper()
    return f"DEMO-{code}-{seq:02d}"


# 每个会话初始化的两条演示订单：一条符合、一条不符合演示退款条件，
# 这样既能演示「核验通过」，也能演示「程序判定不符合仍需人工复核」。
DEMO_ORDER_DRAFTS = [
    ("扫地机器人 X1 Pro", Decimal("2499.00"), "2026-10-10", "已签收",
     REFUND_ELIGIBLE, "演示数据：签收 7 天内且未拆封"),
    ("随行保温杯 500ml", Decimal("129.00"), "2026-09-01", "已签收",
     REFUND_INELIGIBLE, "演示数据：已超过演示退货期且已使用"),
]

_ORDER_FIELDS = ("order_no, product, amount, purchased_at, order_status, "
                 "refund_eligible, eligibility_note")


def ensure_demo_orders(session_id: str) -> list[dict[str, Any]]:
    """返回当前会话的演示订单；首次调用时初始化两条合成订单。

    只查 session_id 对应的记录，不返回其他会话的订单。
    """
    if not session_id:
        return []
    with connect() as conn:
        rows = conn.execute(
            f"SELECT {_ORDER_FIELDS} FROM demo_order WHERE session_id = %s ORDER BY id",
            (session_id,),
        ).fetchall()
        if rows:
            return [dict(row) for row in rows]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for seq, (product, amount, purchased_at, order_status,
                  eligible, note) in enumerate(DEMO_ORDER_DRAFTS, 1):
            conn.execute(
                "INSERT INTO demo_order (order_no, session_id, product, amount, "
                "purchased_at, order_status, refund_eligible, eligibility_note, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (session_id, order_no) DO NOTHING",
                (demo_order_no(session_id, seq), session_id, product, amount,
                 purchased_at, order_status, eligible, note, now),
            )
        rows = conn.execute(
            f"SELECT {_ORDER_FIELDS} FROM demo_order WHERE session_id = %s ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_demo_order(session_id: str, order_no: str) -> dict[str, Any] | None:
    """按订单号查询当前会话的演示订单。

    强制同时匹配 session_id：订单不存在、属于其他会话或不是演示订单，
    统一返回 None，由调用方给出「无法核验」的相同提示，避免泄露订单是否存在。
    """
    # 去掉用户输入里的空格差异，再按大写比对；仍是同会话内查询
    normalized = "".join((order_no or "").split()).upper()
    if not session_id or not normalized:
        return None
    with connect() as conn:
        row = conn.execute(
            f"SELECT {_ORDER_FIELDS} FROM demo_order "
            "WHERE session_id = %s AND UPPER(order_no) = %s",
            (session_id, normalized),
        ).fetchone()
        return dict(row) if row else None


def stats() -> dict[str, Any]:
    """管理后台顶部的统计数字与情绪分布"""
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM chat_log WHERE role = 'user'").fetchone()["c"]
        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) AS c FROM chat_log").fetchone()["c"]
        need_human = conn.execute(
            "SELECT COUNT(*) AS c FROM chat_log WHERE role = 'user' AND need_human = 1"
        ).fetchone()["c"]
        urgent = conn.execute(
            "SELECT COUNT(*) AS c FROM chat_log WHERE role = 'user' AND urgency = '高'"
        ).fetchone()["c"]
        emotions = conn.execute(
            "SELECT emotion, COUNT(*) AS c FROM chat_log "
            "WHERE role = 'user' AND emotion IS NOT NULL GROUP BY emotion ORDER BY c DESC"
        ).fetchall()
        categories = conn.execute(
            "SELECT category, COUNT(*) AS c FROM chat_log "
            "WHERE role = 'user' AND category IS NOT NULL GROUP BY category ORDER BY c DESC"
        ).fetchall()
        return {
            "total": total,
            "sessions": sessions,
            "need_human": need_human,
            "urgent": urgent,
            "emotions": {row["emotion"]: row["c"] for row in emotions},
            "categories": {row["category"]: row["c"] for row in categories},
        }


def list_sessions() -> list[dict[str, Any]]:
    """按会话汇总，供管理后台按会话分组查看与清理"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT session_id, COUNT(*) AS total, "
            "SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_count, "
            "SUM(CASE WHEN need_human = 1 THEN 1 ELSE 0 END) AS need_human_count, "
            "MIN(created_at) AS first_time, MAX(created_at) AS last_time "
            "FROM chat_log WHERE session_id IS NOT NULL "
            "GROUP BY session_id ORDER BY last_time DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def delete_turns(session_id: str | None = None) -> int:
    """删除某个会话的记录；session_id 为 None 时清空全部记录，返回删除条数"""
    with connect() as conn:
        if session_id:
            cursor = conn.execute("DELETE FROM chat_log WHERE session_id = %s", (session_id,))
        else:
            cursor = conn.execute("DELETE FROM chat_log")
        return cursor.rowcount


def to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """导出为带 BOM 的 CSV，便于 Excel 直接识别中文"""
    buffer = io.StringIO()
    buffer.write("\ufeff")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in COLUMNS})
    return buffer.getvalue()


if __name__ == "__main__":
    init_db()
    info = conninfo_to_dict(dsn())
    # 只打印主机/端口/库名，绝不打印密码
    print(f"数据库：{info.get('host')}:{info.get('port')}/{info.get('dbname')}")
    with connect() as conn:
        tables = [row["tablename"] for row in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )]
        for table in tables:
            count = conn.execute(f'SELECT COUNT(*) AS c FROM "{table}"').fetchone()["c"]
            print(f"表 {table}：{count} 行")
    print(f"统计：{stats()}")
