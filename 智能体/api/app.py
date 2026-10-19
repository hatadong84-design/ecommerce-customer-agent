import csv
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from urllib.parse import quote, urlencode

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import ToolMessage
from markdown_it import MarkdownIt
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from 智能体.admin import auth, store
from 智能体.agent.agent import agent
from 智能体.agent.tools.agent_tools import (
    RAG_SOURCE_PREFIX,
    extract_rag_sources,
    set_current_session,
    set_current_ticket,
)
from 智能体.agent.category_service import CATEGORIES, CategoryService
from 智能体.agent.report_service import ReportService
from 智能体.agent.sentiment_service import SentimentService
from 智能体.config_handler import agent_conf
from 智能体.logger_hander import logger
from 智能体.path_toll import get_abs_path
from 智能体.rag import kb_service

ROLES = ["普通用户", "管理员"]
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("缺少 SESSION_SECRET 环境变量，请在项目根目录的 .env 中配置后再启动")

# 会话记忆保留的最近消息条数，避免上下文无限增长
HISTORY_LIMIT = 20

templates = Jinja2Templates(directory=get_abs_path("智能体/api/templates"))

# html=False 会转义原文里的 HTML 标签，避免把用户输入当标签执行
_markdown = MarkdownIt("commonmark", {"html": False, "linkify": False})


def render_markdown(text) -> Markup:
    """把 Markdown 原文渲染成 HTML，返回 Markup 以免被模板二次转义"""
    return Markup(_markdown.render(str(text or "")))


templates.env.filters["md"] = render_markdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    # 首次启动：建默认商品分类，并把已有的 txt 知识导入后台管理器
    kb_service.ensure_kb_seeded()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
app.mount("/static", StaticFiles(directory=get_abs_path("智能体/api/static")), name="static")


@lru_cache(maxsize=1)
def get_report_service() -> ReportService:
    return ReportService()


@lru_cache(maxsize=1)
def get_sentiment_service() -> SentimentService:
    return SentimentService()


@lru_cache(maxsize=1)
def get_category_service() -> CategoryService:
    return CategoryService()


@lru_cache(maxsize=1)
def get_kb_service() -> kb_service.KnowledgeBaseService:
    return kb_service.KnowledgeBaseService()


def is_admin(request: Request) -> bool:
    return bool(request.session.get("admin_user"))


def current_role(request: Request) -> str:
    return request.session.get("role") or ROLES[0]


def session_id_of(request: Request) -> str:
    session = request.session
    user_id = session.get("session_id")
    # 兼容历史会话：旧的随机 ID（如 b5b8890c）一律换成新的「用户NNNN」编号
    if not user_id or not str(user_id).startswith(store.USER_ID_PREFIX):
        session["session_id"] = store.new_user_id()
    return session["session_id"]


def base_context(request: Request) -> dict:
    return {
        "is_admin": is_admin(request),
        "role": current_role(request),
        "admin_user": request.session.get("admin_user"),
        "session_id": session_id_of(request),
        "flash": request.query_params.get("flash") or "",
        "roles": ROLES,
    }


def record(*args, **kwargs) -> None:
    """把一轮对话写入审计表，写库失败不影响对话"""
    try:
        store.log_turn(*args, **kwargs)
    except Exception as e:
        logger.error(f"[record]写入对话记录失败，{str(e)}")


def with_rag_sources(answer: str, sources: list[str]) -> str:
    """把知识库引用来源附到回答末尾（模型已自带来源时不重复追加）"""
    if not sources or RAG_SOURCE_PREFIX in answer:
        return answer
    return f"{answer}\n\n{RAG_SOURCE_PREFIX}{'、'.join(sources)}"


def needs_ticket(payload: dict) -> bool:
    """是否需要创建转人工工单。

    退款类诉求一律交给人工核实（模型不得自行承诺退款结果），
    其余情况仍按情绪分析给出的 need_human 判断。"""
    return bool(payload.get("need_human")) or payload.get("category") == "退款"


def handoff_notice(ticket: dict | None) -> str:
    """工单提交后给用户看的明确告知：已提交、工单号、当前状态、后续由人工跟进。

    直接拼在回答末尾，保证「建单即告知」，不依赖模型是否愿意提这件事。"""
    if not ticket:
        return ""
    verb = "已为您提交" if ticket.get("created") else "已为您更新"
    return (
        f"\n\n【人工工单】{verb}工单 #{ticket['id']}，当前状态：{ticket['status']}。"
        "人工客服会核实您的诉求后跟进处理，具体处理结果请以人工回复为准。"
    )


def with_handoff_notice(answer: str, ticket: dict | None) -> str:
    if not ticket or "【人工工单】" in answer:
        return answer
    return answer + handoff_notice(ticket)


def ask_agent(messages: list[dict]) -> tuple[str, list[str]]:
    """同步跑一遍 agent（带完整会话上下文），返回最终回答与调用过的工具名"""
    answer, tools_used, rag_sources = "", [], []
    for chunk in agent.stream({"messages": messages}, stream_mode="values"):
        message = chunk["messages"][-1]
        if isinstance(message, ToolMessage):
            if message.name == "rag_summarize":
                for source in extract_rag_sources(message.content):
                    if source not in rag_sources:
                        rag_sources.append(source)
            continue
        if getattr(message, "tool_calls", None):
            tools_used.extend(tc["name"] for tc in message.tool_calls)
        elif message.content:
            answer = message.content
    return with_rag_sources(answer, rag_sources), tools_used


def agent_messages(session_id: str) -> list[dict]:
    """取该会话最近的问答作为模型输入，实现多轮记忆（本轮提问已先写入记录）"""
    return store.conversation_history(session_id, limit=HISTORY_LIMIT)


def analyze_question(question: str) -> dict:
    """一次拿到情绪分析与业务分类，合并成入库/展示用的 payload"""
    payload = get_sentiment_service().analyze(question).model_dump()
    payload["category"] = get_category_service().classify(question).category
    return payload


def handle_turn(session_id: str, question: str, payload: dict) -> tuple[str, list[str]]:
    """一轮完整问答：写入用户输入、必要时开转人工工单、调用 agent、写入回答"""
    # 把当前会话身份交给工具层，保证 get_user_id / get_user_location 用的是本会话
    set_current_session(session_id)
    record(session_id, "user", question, payload)
    ticket = None
    try:
        if needs_ticket(payload):
            ticket = store.sync_handoff_ticket(session_id, question, payload)
    except Exception as e:
        logger.error(f"[handle_turn]创建转人工工单失败，{str(e)}")
    # 把本轮工单交给提示词，模型才会用真实工单号；没有工单时清空，避免串到下一轮
    set_current_ticket(ticket)

    answer, tools_used = ask_agent(agent_messages(session_id))
    answer = with_handoff_notice(answer, ticket)
    record(session_id, "assistant", answer, tools=tools_used)
    return answer, tools_used


def load_records_index() -> tuple[list[str], list[str]]:
    """从使用记录中取出可选的用户与月份。

    记录里的用户 ID 是纯数字，全站展示统一加上「用户」前缀（与问答页编号同一写法），
    提交给智能体时也用它，由 fetch_external_data 负责剥掉前缀去比对原始数据。
    """
    path = get_abs_path(agent_conf["external_data_path"])
    users, months = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user = f"{store.USER_ID_PREFIX}{row['用户ID']}"
            if user not in users:
                users.append(user)
            if row["月份"] not in months:
                months.append(row["月份"])
    return sorted(users), sorted(months, reverse=True)


def message_dict(row: dict, admin: bool, tickets: dict[str, dict]) -> dict:
    """把一行 chat_log 转成模板用的消息；非管理员不放入任何分析字段"""
    item = {"role": row["role"], "content": row["content"], "open": False}
    if row["role"] == "user":
        item["session_id"] = row["session_id"]
        # 「已申请人工客服」以工单表为准：只有确实建了工单才提示，
        # 并带上工单号与状态，避免早期数据出现「标了需转人工、却查不到工单」的误导。
        ticket = tickets.get(row["session_id"])
        if ticket:
            item["ticket"] = ticket
    if admin and row["role"] == "user":
        item["sentiment"] = {
            "emotion": row["emotion"] or "",
            "emotion_level": row["emotion_level"],
            "urgency": row["urgency"] or "",
            "need_human": bool(row["need_human"]) if row["need_human"] is not None else False,
            "need_human_reason": row["need_human_reason"] or "",
            "satisfaction": row["satisfaction"] or "",
            "summary": row["summary"] or "",
            "category": row["category"] or "",
        }
    if row["role"] == "assistant" and row["tools_used"]:
        item["tools_used"] = row["tools_used"]
    return item


def build_messages(rows: list[dict], admin: bool) -> list[dict]:
    """问答页的消息列表：管理员把最后一条用户输入的分析默认展开"""
    tickets = store.session_ticket_map()
    messages = [message_dict(row, admin, tickets) for row in rows]
    if admin:
        for item in reversed(messages):
            if item["role"] == "user":
                item["open"] = True
                break
    return messages


def build_threads(rows: list[dict], admin: bool) -> list[dict]:
    """管理后台的消息列表：按会话分成一段段对话（rows 为按时间倒序）"""
    threads: list[dict] = []
    index: dict[str, dict] = {}
    tickets = store.session_ticket_map()
    for row in rows:
        sid = row["session_id"] or "（无会话）"
        thread = index.get(sid)
        if thread is None:
            thread = {"session_id": sid, "total": 0, "last_time": row["created_at"],
                      "messages": [], "ticket": tickets.get(row["session_id"])}
            index[sid] = thread
            threads.append(thread)
        thread["messages"].append(message_dict(row, admin, tickets))
        thread["total"] += 1
    for thread in threads:
        thread["messages"].reverse()  # 每段会话内改为正序
    return threads


def filter_kwargs(session_id: str, keyword: str, urgency: str, human: int,
                  category: str = "全部") -> dict:
    return {
        "session_id": session_id or None,
        "keyword": (keyword or "").strip() or None,
        "urgency": None if urgency in ("", "全部") else urgency,
        "need_human": True if human else None,
        "category": None if category in ("", "全部") else category,
    }


@app.get("/")
def index(request: Request):
    rows = store.query_turns(session_id=session_id_of(request), limit=500)
    rows.reverse()  # 查询是倒序，展示改为正序
    context = base_context(request)
    context["messages"] = build_messages(rows, context["is_admin"])
    # 管理员为「本用户」生成过的报告，直接展示在他自己的问答页里
    context["my_reports"] = store.list_usage_reports(context["session_id"])
    return templates.TemplateResponse(request, "index.html", context)


@app.post("/chat")
def chat(request: Request, question: str = Form(...)):
    """无 JS 时的回退路径：阻塞跑完一轮问答后整页刷新"""
    question = (question or "").strip()
    if not question:
        return RedirectResponse("/", status_code=303)
    handle_turn(session_id_of(request), question, analyze_question(question))
    return RedirectResponse("/", status_code=303)


def sse(event: str, data) -> str:
    """拼一条 Server-Sent Events 消息"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_turn(session_id: str, question: str, admin: bool):
    """流式返回一轮问答：先回显提问与情绪分析，再推工具调用提示与最终回答"""
    # 先把会话身份交给工具层；langchain 的 run_in_executor 会 copy_context，
    # 因此即使工具在子线程执行，也能读到这里的当前会话 ID。
    set_current_session(session_id)
    try:
        payload = await run_in_threadpool(analyze_question, question)

        await run_in_threadpool(record, session_id, "user", question, payload)
        ticket = None
        if needs_ticket(payload):
            ticket = await run_in_threadpool(store.sync_handoff_ticket, session_id, question, payload)
        # 把本轮工单交给提示词，模型才会用真实工单号；没有工单时清空，避免串到下一轮
        set_current_ticket(ticket)

        # 分析字段仅管理员可见，普通用户只收到「是否转人工」与工单信息
        analysis_html = ""
        if admin:
            macro = templates.get_template("_analysis.html").module.render_analysis
            analysis_html = str(macro(payload, True, ticket, session_id))
        yield sse("sentiment", {
            "need_human": payload["need_human"],
            "ticket": ticket,
            "analysis_html": analysis_html if admin else "",
        })

        messages = await run_in_threadpool(agent_messages, session_id)
        texts, tools_used, rag_sources = [], [], []
        async for chunk, metadata in agent.astream({"messages": messages}, stream_mode="messages"):
            # 知识库检索的引用来源随工具输出一起到达，收集起来稍后附到回答末尾
            if isinstance(chunk, ToolMessage):
                if chunk.name == "rag_summarize":
                    for source in extract_rag_sources(chunk.content):
                        if source not in rag_sources:
                            rag_sources.append(source)
                continue
            # 只转发 agent 节点自己的输出，跳过工具内部的中间模型输出
            if metadata.get("langgraph_node") != "agent":
                continue
            for call in getattr(chunk, "tool_calls", None) or []:
                name = call.get("name")
                if name and name not in tools_used:
                    tools_used.append(name)
                    yield sse("tool", name)
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                texts.append(content)
                yield sse("token", content)

        answer = "".join(texts).strip() or "抱歉，我暂时没能理解这个问题，可以换个说法再问一次吗？"
        answer = with_rag_sources(answer, rag_sources)
        answer = with_handoff_notice(answer, ticket)
        await run_in_threadpool(record, session_id, "assistant", answer, None, tools_used)
        yield sse("done", {"html": str(render_markdown(answer)), "tools_used": tools_used})
    except Exception as e:
        logger.error(f"[stream_turn]流式对话失败，{str(e)}")
        yield sse("error", "服务暂时不可用，请稍后重试")


@app.post("/chat/stream")
async def chat_stream(request: Request, question: str = Form(...)):
    session_id = session_id_of(request)
    question = (question or "").strip()
    if not question:
        return Response(status_code=204)
    return StreamingResponse(
        stream_turn(session_id, question, is_admin(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/report")
def report_page(request: Request):
    # 使用报告含用户设备数据，只对管理员开放
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    users, months = load_records_index()
    context = base_context(request)
    context.update({"users": users, "months": months, "report": None,
                    "selected_user": users[0], "selected_month": months[0],
                    "saved_months": [r["month"] for r in store.list_usage_reports(users[0])]})
    return templates.TemplateResponse(request, "report.html", context)


@app.post("/report")
def report_generate(request: Request, user_id: str = Form(...), month: str = Form(...)):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    users, months = load_records_index()
    report = get_report_service().generate_report(user_id, month)
    # 生成即落库，用户在自己问答页就能看到；同一用户同一月份重复生成会覆盖
    if report:
        store.save_usage_report(user_id, month, report)
    context = base_context(request)
    context.update({"users": users, "months": months, "report": report,
                    "selected_user": user_id, "selected_month": month,
                    "saved_months": [r["month"] for r in store.list_usage_reports(user_id)]})
    return templates.TemplateResponse(request, "report.html", context)


@app.post("/role")
def switch_role(request: Request, role: str = Form(...)):
    request.session["role"] = role if role in ROLES else ROLES[0]
    return RedirectResponse("/", status_code=303)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if auth.authenticate(username, password):
        request.session["admin_user"] = username.strip()
        request.session["role"] = ROLES[1]
        return RedirectResponse("/", status_code=303)
    return RedirectResponse(f"/?flash={quote('用户名或密码错误')}", status_code=303)


@app.post("/register")
def register(request: Request, username: str = Form(...), password: str = Form(...),
             invite_code: str = Form(...)):
    ok, message = auth.register(username, password, invite_code)
    if ok:
        request.session["admin_user"] = username.strip()
        request.session["role"] = ROLES[1]
        return RedirectResponse("/", status_code=303)
    return RedirectResponse(f"/?flash={quote(message)}", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.pop("admin_user", None)
    request.session["role"] = ROLES[0]
    return RedirectResponse("/", status_code=303)


@app.get("/admin")
def admin_page(request: Request, session_id: str = "",
               q: str = "", urgency: str = "全部", human: int = 0,
               category: str = "全部"):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)

    sessions = store.list_sessions()
    rows = store.query_turns(**filter_kwargs(session_id, q, urgency, human, category))

    context = base_context(request)
    context.update({
        "stats": store.stats(),
        "ticket_counts": store.ticket_counts(),
        "sessions": sessions,
        "rows": rows,
        "threads": build_threads(rows, True),
        "selected_session": session_id,
        "q": q,
        "urgency": urgency,
        "human": human,
        "category": category,
        "categories": CATEGORIES,
        "export_url": "/admin/export.csv?" + urlencode(
            {"session_id": session_id, "q": q, "urgency": urgency, "human": human,
             "category": category}
        ),
    })
    return templates.TemplateResponse(request, "admin.html", context)


@app.get("/admin/export.csv")
def admin_export(request: Request, session_id: str = "", q: str = "",
                 urgency: str = "全部", human: int = 0, category: str = "全部"):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    rows = store.query_turns(**filter_kwargs(session_id, q, urgency, human, category))
    return Response(
        content=store.to_csv(rows),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="chat_log.csv"'},
    )


@app.post("/admin/cleanup")
def admin_cleanup(request: Request, target: str = Form("")):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    deleted = store.delete_turns(target or None)
    return RedirectResponse(f"/admin?flash={quote(f'已删除 {deleted} 条记录')}", status_code=303)


@app.get("/admin/tickets")
def admin_tickets(request: Request, status: str = "全部", session_id: str = ""):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    current = None if status in ("", "全部") else status
    context = base_context(request)
    context.update({
        "tickets": store.query_tickets(current, session_id or None),
        "counts": store.ticket_counts(),
        "selected_status": status,
        "selected_session": session_id,
        "statuses": store.TICKET_STATUSES,
    })
    return templates.TemplateResponse(request, "tickets.html", context)


@app.post("/admin/tickets/update")
def admin_ticket_update(request: Request, ticket_id: int = Form(...), status: str = Form(...),
                        agent_name: str = Form(""), resolution: str = Form(""),
                        close_reason: str = Form("")):
    """推进工单状态，同时可记录责任人、处理记录与关闭原因"""
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    if status in store.TICKET_STATUSES:
        store.set_ticket_status(
            ticket_id, status,
            (agent_name or "").strip() or None,
            (resolution or "").strip() or None,
            (close_reason or "").strip() or None,
        )
    return RedirectResponse(f"/admin/tickets?flash={quote('工单状态已更新')}", status_code=303)


# ============================================================
# 知识库管理：按商品分类维护知识文档（上传 txt / pdf / word，并支持改删）
# ------------------------------------------------------------
# 每个商品分类 = 一份独立的「数据管理器」，页面 /admin/kb/{分类}
# ============================================================
def kb_redirect(category: str, message: str) -> RedirectResponse:
    """回到某个分类的数据管理器并带上提示"""
    return RedirectResponse(
        f"/admin/kb/{quote(category, safe='')}?flash={quote(message)}", status_code=303
    )


@app.get("/admin/kb")
def kb_index(request: Request):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    context = base_context(request)
    context.update({
        "categories": store.list_kb_categories(),
        "total": store.kb_document_total(),
    })
    return templates.TemplateResponse(request, "kb.html", context)


@app.get("/admin/kb/{category}")
def kb_category_page(request: Request, category: str):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    context = base_context(request)
    context.update({
        "category": category,
        "documents": store.list_kb_documents(category, with_content=True),
    })
    return templates.TemplateResponse(request, "kb_category.html", context)


@app.post("/admin/kb/cat/create")
def kb_category_create(request: Request, name: str = Form(...)):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    name = (name or "").strip()
    if not name:
        return RedirectResponse(f"/admin/kb?flash={quote('分类名称不能为空')}", status_code=303)
    if store.kb_category_exists(name):
        return RedirectResponse(f"/admin/kb?flash={quote(f'分类「{name}」已存在')}", status_code=303)
    store.create_kb_category(name)
    return RedirectResponse(f"/admin/kb/{quote(name, safe='')}", status_code=303)


@app.post("/admin/kb/cat/delete")
def kb_category_delete(request: Request, name: str = Form(...)):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    removed = get_kb_service().delete_category(name)
    return RedirectResponse(
        f"/admin/kb?flash={quote(f'已删除分类「{name}」及其 {removed} 篇文档')}", status_code=303
    )


@app.post("/admin/kb/doc/upload")
async def kb_document_upload(request: Request, category: str = Form(...),
                             title: str = Form(""), file: UploadFile = File(...)):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)

    data = await file.read()
    filename = file.filename or ""
    try:
        text, source_type = kb_service.parse_upload(filename, data)
    except ValueError as e:
        return kb_redirect(category, str(e))

    title = (title or "").strip() or os.path.splitext(filename)[0]
    await run_in_threadpool(get_kb_service().add_document, category, title, text, source_type)
    return kb_redirect(category, f"已上传《{title}》（{source_type}，{len(text)} 字）并建立索引")


@app.post("/admin/kb/doc/update")
async def kb_document_update(request: Request, doc_id: int = Form(...),
                             title: str = Form(...), content: str = Form(...)):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    row = await run_in_threadpool(
        get_kb_service().update_document, doc_id, title.strip(), content
    )
    if not row:
        return RedirectResponse(f"/admin/kb?flash={quote('文档不存在')}", status_code=303)
    return kb_redirect(row["category"], f"已保存《{row['title']}》并重新建立索引")


@app.post("/admin/kb/doc/delete")
async def kb_document_delete(request: Request, doc_id: int = Form(...)):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    row = await run_in_threadpool(get_kb_service().delete_document, doc_id)
    if not row:
        return RedirectResponse(f"/admin/kb?flash={quote('文档不存在')}", status_code=303)
    return kb_redirect(row["category"], f"已删除《{row['title']}》并清理其向量分片")


@app.post("/admin/kb/doc/reindex")
async def kb_document_reindex(request: Request, doc_id: int = Form(...)):
    """对单篇文档重试建索引（索引失败后修好数据源，再点重试即可）"""
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    row = await run_in_threadpool(get_kb_service().reindex_document, doc_id)
    if not row:
        return RedirectResponse(f"/admin/kb?flash={quote('文档不存在')}", status_code=303)
    if row.get("index_status") == store.KB_INDEX_DONE:
        return kb_redirect(row["category"], f"《{row['title']}》已重新建立索引")
    return kb_redirect(
        row["category"],
        f"《{row['title']}》建立索引仍失败：{row.get('index_error') or '未知原因'}",
    )


@app.post("/admin/kb/rebuild")
async def kb_rebuild(request: Request):
    if not is_admin(request):
        return RedirectResponse("/", status_code=303)
    count = await run_in_threadpool(get_kb_service().rebuild_index)
    return RedirectResponse(f"/admin/kb?flash={quote(f'向量库已按 {count} 篇文档重建')}", status_code=303)


if __name__ == "__main__":
    uvicorn.run("智能体.api.app:app", host="127.0.0.1", port=8000)
