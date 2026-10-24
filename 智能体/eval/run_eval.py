# -*- coding: utf-8 -*-
"""电商智能助手评测脚本（第一版：表格化记录，不引入评测框架）。

覆盖四类检查：
    retrieval  检索命中正确文档 / 低相关资料被拦下 / 商品分类隔离（含改进前后对比）
    e2e        端到端问答：工具调用、引用来源、无依据却肯定作答
    identity   多会话查询使用记录时是否始终取到自己的用户ID
    kb         上传限制、索引失败与重试/重建流程

用法（cwd = 项目根 d:\\PythonProject3）：
    .venv\\Scripts\\python.exe 智能体\\eval\\run_eval.py all
    .venv\\Scripts\\python.exe 智能体\\eval\\run_eval.py retrieval
结果写入 智能体/eval/results/
"""
import argparse
import asyncio
import csv
import io
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, ToolMessage

from 智能体.admin import store
from 智能体.agent.agent import agent
from 智能体.agent.answer_guard import apply_guard, prefetch_knowledge, should_call_agent
from 智能体.agent.tools.agent_tools import (
    RAG_SOURCE_PREFIX,
    extract_rag_sources,
    set_current_knowledge,
    set_current_session,
)
from 智能体.rag.rag_service import NO_KNOWLEDGE_ANSWER
from 智能体.config_handler import chroma_conf
from 智能体.rag import kb_service
from 智能体.rag.rag_service import RELEVANCE_THRESHOLD, RagSummarizeService
from 智能体.rag.vector_store import VectorStoreService

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
K = chroma_conf["k"]

# 回答里出现这些说法，说明模型明确承认「资料里没有」或「超出服务范围」。
# 分两类：硬拒绝（能力/范围）本身就是正确拒答；软拒绝（没资料）还要看后面有没有补常识。
HARD_REFUSAL_PATTERNS = (
    r"不在服务范围", r"服务范围仅限", r"不属于本助手", r"不属于.*服务范畴",
    r"没有.*功能", r"无法查询", r"无法提供", r"无法帮您", r"不涉及",
    r"不提供", r"不掌握", r"仅提供", r"只能提供", r"超出.*范围",
)
SOFT_REFUSAL_PATTERNS = (
    r"没有找到", r"没有关于", r"未提及", r"没有提及", r"没有相关信息", r"暂无相关",
    r"资料中没有", r"知识库中没有", r"没有收录", r"未收录",
)

# 「拒答之后又给结论」的信号：只认实质性判断，不把「建议补充型号/转人工」当成作答
GENERIC_AFTER_REFUSAL_PATTERNS = (
    r"通常", r"一般", r"取决于", r"大概", r"往往", r"常见的是",
    r"为准", r"需根据", r"综合判定", r"视.*而定", r"通常来说",
)

# 没有调用检索，却在回答里声称有依据（含编造「参考来源」）
CLAIMED_RETRIEVAL_PATTERNS = (
    r"检索结果", r"根据知识库", r"根据资料", r"专业知识库", r"检索到", r"从知识库中",
    r"参考来源", r"依据.*(规范|标准|资料|知识库)",
)


def _first_match(patterns, text: str):
    matches = [m for p in patterns for m in re.finditer(p, text)]
    return min(matches, key=lambda m: m.start()) if matches else None


def refusal_hit(answer: str) -> bool:
    """回答是否包含明确拒绝（没资料或超出服务范围）"""
    text = str(answer or "")
    return _first_match(HARD_REFUSAL_PATTERNS, text) is not None \
        or _first_match(SOFT_REFUSAL_PATTERNS, text) is not None


def substantive_after_refusal(answer: str) -> bool:
    """先承认没资料（或超出范围）、随后又给出实质性结论：这是本轮要防的失败模式"""
    text = str(answer or "")
    first = _first_match(HARD_REFUSAL_PATTERNS, text) \
        or _first_match(SOFT_REFUSAL_PATTERNS, text)
    if not first:
        return False
    tail = text[first.end():]
    if len(tail) <= 40:
        return False
    return any(re.search(p, tail) for p in GENERIC_AFTER_REFUSAL_PATTERNS)


def claimed_retrieval_without_tool(answer: str, called_rag: bool,
                                   preretrieved: bool = False) -> bool:
    """回答声称「知识库检索结果」，但本轮既没调工具也没预检索"""
    if called_rag or preretrieved:
        return False
    return _first_match(CLAIMED_RETRIEVAL_PATTERNS, str(answer or "")) is not None


def sources_in_answer(answer: str) -> list[str]:
    """从最终回答的「参考来源：」后解析文档标题"""
    text = str(answer or "")
    index = text.find(RAG_SOURCE_PREFIX)
    if index < 0:
        return []
    tail = text[index + len(RAG_SOURCE_PREFIX):].splitlines()
    if not tail:
        return []
    return [name.strip() for name in tail[0].split("、") if name.strip()]


def load_questions() -> list[dict]:
    with open(HERE / "questions.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dump(name: str, fieldnames: list[str], rows: list[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def titles_of(docs) -> str:
    return ";".join((d.metadata.get("title") or "?") for d in docs[:3])


# ============================================================
# 1. 检索层：命中 / 拦截 / 分类隔离
# ============================================================
def retrieval_layer() -> None:
    svc = RagSummarizeService()
    chroma = svc.vector_store.vector_store
    rows: list[dict] = []

    for q in load_questions():
        # 多轮的第 2 轮脱离上下文检索会天然失准，不计入检索层统计
        if q["turn"] != "1":
            continue
        question = q["question"]
        expect_docs = [t for t in q["expect_docs"].split("|") if t.strip()]
        expect_cat = q["expect_category"].strip()

        # 改进前：无相关度门槛、无商品分类过滤
        before = chroma.similarity_search(question, k=K)
        # 改进后：带分类过滤 + 相关度门槛
        passed = [(d, s) for d, s in svc.vector_store.retrieve_with_scores(
            question, category=expect_cat or None) if s >= RELEVANCE_THRESHOLD]
        # 分类过滤若被绕过（模型没传 category）时的污染情况
        no_filter = [d for d, s in svc.vector_store.retrieve_with_scores(question)
                     if s >= RELEVANCE_THRESHOLD]

        def contamination(docs) -> float | str:
            if not expect_cat or not docs:
                return ""
            bad = sum(1 for d in docs if d.metadata.get("category") != expect_cat)
            return round(bad / len(docs), 2)

        docs_after = [d for d, _ in passed]
        hit_before = hit_after = ""
        if expect_docs:
            hit_before = any((d.metadata.get("title") or "") in expect_docs for d in before)
            hit_after = any((d.metadata.get("title") or "") in expect_docs for d in docs_after)

        gated_before = gated_after = ""
        if q["expect_behavior"] == "明确拒答":
            gated_before = len(before) == 0
            gated_after = len(passed) == 0

        rows.append({
            "id": q["id"], "group": q["group"], "question": question,
            "expect_docs": q["expect_docs"], "expect_category": expect_cat,
            "before_pass": len(before), "before_titles": titles_of(before),
            "after_pass": len(docs_after), "after_titles": titles_of(docs_after),
            "after_top_score": round(passed[0][1], 4) if passed else "",
            "hit_before": hit_before, "hit_after": hit_after,
            "contam_after": contamination(docs_after), "contam_no_filter": contamination(no_filter),
            "gated_before": gated_before, "gated_after": gated_after,
        })

    fields = ["id", "group", "question", "expect_docs", "expect_category", "before_pass",
              "before_titles", "after_pass", "after_titles", "after_top_score",
              "hit_before", "hit_after", "contam_after", "contam_no_filter",
              "gated_before", "gated_after"]
    dump("retrieval_results.csv", fields, rows)

    def ratio(items, key) -> tuple[int, int]:
        values = [r[key] for r in items if r[key] != ""]
        return sum(1 for v in values if v), len(values)

    hit_rows = [r for r in rows if r["expect_docs"]]
    gate_rows = [r for r in rows if r["gated_before"] != ""]
    cat_rows = [r for r in rows if r["expect_category"]]

    def mean(items, key) -> float:
        values = [r[key] for r in items if r[key] != ""]
        return round(sum(values) / len(values), 3) if values else 0.0

    summary = [
        {"metric": "检索命中率", "before": f"{ratio(hit_rows, 'hit_before')[0]}/{ratio(hit_rows, 'hit_before')[1]}",
         "after": f"{ratio(hit_rows, 'hit_after')[0]}/{ratio(hit_rows, 'hit_after')[1]}"},
        {"metric": "低相关拦截率", "before": f"{ratio(gate_rows, 'gated_before')[0]}/{ratio(gate_rows, 'gated_before')[1]}",
         "after": f"{ratio(gate_rows, 'gated_after')[0]}/{ratio(gate_rows, 'gated_after')[1]}"},
        {"metric": "平均跨分类污染率", "before": "",
         "after": f"{mean(cat_rows, 'contam_after')}（不过滤时 {mean(cat_rows, 'contam_no_filter')}）"},
        {"metric": "平均放行条数", "before": round(sum(r["before_pass"] for r in rows) / len(rows), 2),
         "after": round(sum(r["after_pass"] for r in rows) / len(rows), 2)},
    ]
    dump("retrieval_summary.csv", ["metric", "before", "after"], summary)
    print("[检索层] 共", len(rows), "题")
    for item in summary:
        print(f"  {item['metric']}: 改进前={item['before']} 改进后={item['after']}")


# ============================================================
# 2. 端到端：工具调用 / 来源引用 / 无依据作答
# ============================================================
async def run_agent_turn(session_id: str, history: list[dict]):
    set_current_session(session_id)
    texts, tools, sources = [], [], []
    async for chunk, metadata in agent.astream({"messages": history}, stream_mode="messages"):
        if isinstance(chunk, ToolMessage):
            if chunk.name == "rag_summarize":
                for source in extract_rag_sources(chunk.content):
                    if source not in sources:
                        sources.append(source)
            continue
        if metadata.get("langgraph_node") != "agent":
            continue
        for call in getattr(chunk, "tool_calls", None) or []:
            name = call.get("name")
            if name and name not in tools:
                tools.append(name)
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            texts.append(content)
    return "".join(texts).strip(), tools, sources


async def guard_turn(session_id: str, history: list[dict], question: str):
    """与 Web 路径一致的评测入口：预检索 → 可能短路 → agent → 答案守卫。

    返回 (最终回答, 工具, 工具来源, 预检索结果, 是否被守卫拒答)。
    """
    recent = [m["content"] for m in history if m["role"] == "user"][-4:]
    knowledge = prefetch_knowledge(question, recent_questions=recent)
    set_current_knowledge(knowledge)

    if not should_call_agent(knowledge):
        return NO_KNOWLEDGE_ANSWER, [], [], knowledge, True

    raw, tools, sources = await run_agent_turn(
        session_id, history + [{"role": "user", "content": question}]
    )
    guarded = apply_guard(raw, knowledge)
    refused = guarded == NO_KNOWLEDGE_ANSWER and raw != NO_KNOWLEDGE_ANSWER
    return guarded, tools, sources, knowledge, refused


async def e2e_layer() -> None:
    cases: dict[str, list[dict]] = {}
    for q in load_questions():
        cases.setdefault(q["case_id"], []).append(q)

    rows, answers = [], []
    for index, (case_id, turns) in enumerate(cases.items(), start=1):
        session_id = f"用户9{index:03d}"
        history: list[dict] = []
        for turn in sorted(turns, key=lambda r: int(r["turn"])):
            start = time.time()
            answer, tools, sources, knowledge, guard_refused = await guard_turn(
                session_id, history, turn["question"]
            )
            seconds = round(time.time() - start, 1)
            history += [{"role": "user", "content": turn["question"]},
                        {"role": "assistant", "content": answer}]

            expect_docs = [t for t in turn["expect_docs"].split("|") if t.strip()]
            called_rag = "rag_summarize" in tools
            final_sources = sources_in_answer(answer)
            refusal = refusal_hit(answer)
            # 退款等业务分类也照常预检索：只要推断出商品分类就查资料
            preretrieved = bool(knowledge.get("category"))
            cl = knowledge.get("checked")
            retrieval_used = called_rag or preretrieved
            claimed = claimed_retrieval_without_tool(answer, called_rag, preretrieved)
            rows.append({
                "id": turn["id"], "group": turn["group"], "turn": turn["turn"],
                "session_id": session_id, "question": turn["question"],
                "expect_docs": turn["expect_docs"], "expect_behavior": turn["expect_behavior"],
                "tools_used": ";".join(tools), "called_rag": called_rag,
                "preretrieved": preretrieved, "retrieval_used": retrieval_used,
                "has_source": bool(final_sources), "sources": ";".join(sources),
                "final_sources": ";".join(final_sources),
                "no_info_signal": refusal,
                "flag_miss_tool": bool(expect_docs) and not retrieval_used,
                "flag_no_source": bool(expect_docs) and not final_sources,
                "flag_unsupported": (
                    turn["expect_behavior"] == "明确拒答"
                    and (not refusal or substantive_after_refusal(answer))
                ),
                "claimed_retrieval_without_tool": claimed,
                "guard_refused": guard_refused,
                "guard_refused_on_answerable": bool(expect_docs) and guard_refused,
                "preretrieval_checked": cl,
                "preretrieval_category": knowledge.get("category") or "",
                "preretrieval_reason": knowledge.get("reason") or "",
                "seconds": seconds,
                "answer": answer[:200].replace("\n", " "),
            })
            answers.append(f"## {turn['id']}（{turn['group']}）{turn['question']}\n"
                           f"tools={tools} sources={sources} final_sources={final_sources} "
                           f"preretrieved={preretrieved}/{knowledge.get('category') or '无分类'} "
                           f"guard_refused={guard_refused} claimed_without_tool={claimed} "
                           f"seconds={seconds}\n\n{answer}\n")

    fields = ["id", "group", "turn", "session_id", "question", "expect_docs", "expect_behavior",
              "tools_used", "called_rag", "preretrieved", "preretrieval_category",
              "preretrieval_checked", "preretrieval_reason", "retrieval_used",
              "has_source", "sources", "final_sources", "no_info_signal",
              "flag_miss_tool", "flag_no_source", "flag_unsupported",
              "claimed_retrieval_without_tool", "guard_refused",
              "guard_refused_on_answerable", "seconds", "answer"]
    dump("e2e_results.csv", fields, rows)
    (RESULTS / "answers.md").write_text("\n".join(answers), encoding="utf-8")

    kb_rows = [r for r in rows if r["expect_docs"]]
    refuse_rows = [r for r in rows if r["expect_behavior"] == "明确拒答"]
    print(f"[端到端] 共 {len(rows)} 轮，平均 {round(sum(r['seconds'] for r in rows) / len(rows), 1)} 秒/轮")
    print(f"  应检索但无依据 {sum(1 for r in kb_rows if r['flag_miss_tool'])}/{len(kb_rows)}")
    print(f"  命中预期但缺来源 {sum(1 for r in kb_rows if r['flag_no_source'])}/{len(kb_rows)}")
    print(f"  无答案题疑似无依据作答 {sum(1 for r in refuse_rows if r['flag_unsupported'])}/{len(refuse_rows)}")
    for r in refuse_rows:
        if r["flag_unsupported"]:
            print(f"    仍标记：id {r['id']} {r['question']}")
    print(f"  声称检索但未检索 "
          f"{sum(1 for r in rows if r['claimed_retrieval_without_tool'])}/{len(rows)}")
    print(f"  有资料却被拒答 "
          f"{sum(1 for r in rows if r['guard_refused_on_answerable'])}/{len(kb_rows)}")
    for r in rows:
        if r["guard_refused"]:
            print(f"    守卫拒答：id {r['id']} 分类={r['preretrieval_category'] or '无'} "
                  f"原因={r['preretrieval_reason']}")
    write_e2e_summary(rows)


def _true(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _count(rows: list[dict], key: str) -> int:
    return sum(1 for r in rows if _true(r.get(key)))


def _rate(rows: list[dict], key: str, total: int) -> str:
    return f"{_count(rows, key)}/{total}" if total else "0/0"


def write_e2e_summary(after: list[dict]) -> None:
    """对照 e2e_results_before.csv 输出 before/after 明细，避免只报一个总数"""
    before: list[dict] = []
    path = RESULTS / "e2e_results_before.csv"
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            before = list(csv.DictReader(f))

    def block(rows: list[dict]) -> dict:
        kb_rows = [r for r in rows if r.get("expect_docs")]
        refuse_rows = [r for r in rows if r.get("expect_behavior") == "明确拒答"]
        return {
            "应检索但无依据": _rate(kb_rows, "flag_miss_tool", len(kb_rows)),
            "命中预期但缺来源": _rate(kb_rows, "flag_no_source", len(kb_rows)),
            "无答案题疑似作答": _rate(refuse_rows, "flag_unsupported", len(refuse_rows)),
            "声称检索但未检索": _rate(rows, "claimed_retrieval_without_tool", len(rows)),
            "有资料却被拒答": _rate(kb_rows, "guard_refused_on_answerable", len(kb_rows)),
            "平均耗时(秒/轮)": round(sum(float(r["seconds"]) for r in rows) / len(rows), 1)
            if rows else 0,
        }

    before_stats, after_stats = block(before), block(after)
    notes = {
        "应检索但无依据": "expect_docs 且本轮既未调工具也未预检索",
        "命中预期但缺来源": "expect_docs 且最终回答没有「参考来源：」",
        "无答案题疑似作答": "明确拒答题命中拒答后又给出通用结论",
        "声称检索但未检索": "回答声称有检索依据，但本轮没有检索动作",
        "有资料却被拒答": "预检索有资料却被守卫拒答，用于发现误拒",
        "平均耗时(秒/轮)": "守卫多一次模型调用，用于衡量延迟代价",
    }
    rows = [{"metric": k, "before": before_stats[k], "after": after_stats[k],
             "note": notes[k]} for k in notes]
    dump("e2e_summary.csv", ["metric", "before", "after", "note"], rows)
    print("[端到端对照] 已写入 e2e_summary.csv")
    for row in rows:
        print(f"  {row['metric']}: 改进前={row['before']} 改进后={row['after']}")


# ============================================================
# 3. 身份一致性：多会话查使用记录
# ============================================================
async def identity_layer() -> None:
    rows = []
    for sid in ["用户1001", "用户1002", "用户1003", "用户1004", "用户1005", "用户1006"]:
        set_current_session(sid)
        result = await agent.ainvoke({"messages": [
            {"role": "user", "content": "帮我查一下我 2026-08 的扫地机器人使用记录"}
        ]})
        calls = [c for m in result["messages"] if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
        fetch = [c for c in calls if c["name"] == "fetch_external_data"]
        args = (fetch[0].get("args") or {}) if fetch else {}
        rows.append({
            "session_id": sid,
            "called_tools": ";".join(sorted({c["name"] for c in calls})),
            "fetch_called": bool(fetch),
            "user_id_arg": str(args.get("user_id", "")),
            "month_arg": str(args.get("month", "")),
            "id_match": (args.get("user_id") == sid) if fetch else "",
        })
    dump("identity_results.csv", ["session_id", "called_tools", "fetch_called",
                                 "user_id_arg", "month_arg", "id_match"], rows)
    matched = sum(1 for r in rows if r["id_match"] is True)
    print(f"[身份一致性] 取到自己ID {matched}/{len(rows)}，未调用工具 "
          f"{sum(1 for r in rows if not r['fetch_called'])}")


# ============================================================
# 4. 知识库流程：上传限制 / 索引失败 / 重试 / 重建
# ============================================================
def many_page_pdf(pages: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def kb_layer() -> None:
    rows: list[dict] = []

    def add(scene: str, expect: str, actual: str, ok: bool) -> None:
        rows.append({"scene": scene, "expect": expect, "actual": actual, "pass": bool(ok)})

    # 上传限制
    cases = [
        ("上传-超过体积", "文件过大", "big.txt", b"a" * (kb_service.MAX_UPLOAD_BYTES + 1)),
        ("上传-不支持的扩展名", "不支持的文件类型", "a.csv", b"x,y,z"),
        ("上传-空内容", "没有解析到有效文本", "empty.txt", b"   "),
        ("上传-PDF页数超限", "PDF 页数过多", "many.pdf",
         many_page_pdf(kb_service.MAX_PDF_PAGES + 1)),
    ]
    for scene, expect, filename, data in cases:
        try:
            kb_service.parse_upload(filename, data)
            add(scene, expect, "未拦截", False)
        except ValueError as e:
            add(scene, expect, str(e), expect in str(e))

    # 正常上传
    try:
        text, source_type = kb_service.parse_upload(
            "ok.txt", "扫地机器人评测临时知识：本机支持定时清扫。".encode("utf-8"))
        add("上传-正常文件", "解析成功不报错", f"{source_type}，{len(text)} 字", len(text) > 0)
    except ValueError as e:
        add("上传-正常文件", "解析成功不报错", str(e), False)

    # 索引失败 → 状态落库 → 重试成功
    service = kb_service.KnowledgeBaseService()
    original = VectorStoreService.index_document

    def broken_index(self, doc_id, category, title, text):
        raise RuntimeError("评测注入的索引失败")

    VectorStoreService.index_document = broken_index
    doc_id = None
    try:
        doc_id = service.add_document("评测临时分类", "评测临时文档",
                                      "评测临时内容：仅用于验证索引失败与重试流程。", "手动")
        failed = store.get_kb_document(doc_id)
    finally:
        VectorStoreService.index_document = original

    add("索引失败-状态记录", store.KB_INDEX_FAILED,
        (failed or {}).get("index_status") or "无记录",
        (failed or {}).get("index_status") == store.KB_INDEX_FAILED)
    add("索引失败-原因记录", "含注入的失败原因", (failed or {}).get("index_error") or "无记录",
        "评测注入的索引失败" in ((failed or {}).get("index_error") or ""))

    retried = service.reindex_document(doc_id)
    add("索引失败-重试建索引", store.KB_INDEX_DONE,
        (retried or {}).get("index_status") or "无记录",
        (retried or {}).get("index_status") == store.KB_INDEX_DONE)

    count = service.rebuild_index()
    pending = [d for d in store.list_kb_documents() if d["index_status"] != store.KB_INDEX_DONE]
    add("全量重建", "全部已索引", f"重建 {count} 篇，异常 {len(pending)} 篇", not pending)

    service.delete_category("评测临时分类")
    names = [c["name"] for c in store.list_kb_categories()]
    add("清理临时分类", "临时分类已删除", f"当前分类：{names}", "评测临时分类" not in names)

    dump("kb_flow_results.csv", ["scene", "expect", "actual", "pass"], rows)
    passed = sum(1 for r in rows if r["pass"])
    print(f"[知识库流程] 通过 {passed}/{len(rows)}")
    for r in rows:
        if not r["pass"]:
            print(f"  未通过：{r['scene']} 期望={r['expect']} 实际={r['actual']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layer", nargs="?", default="all",
                        choices=["retrieval", "e2e", "identity", "kb", "all"])
    args = parser.parse_args()

    if args.layer in ("retrieval", "all"):
        retrieval_layer()
    if args.layer in ("kb", "all"):
        kb_layer()
    if args.layer in ("identity", "all"):
        asyncio.run(identity_layer())
    if args.layer in ("e2e", "all"):
        asyncio.run(e2e_layer())
    print("评测完成，结果见 智能体/eval/results/")


if __name__ == "__main__":
    main()
