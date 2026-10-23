import csv
import os
import sys

import streamlit as st

# streamlit run 智能体/app.py 时 sys.path[0] 是脚本所在目录，这里补上项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from 智能体.admin import ui as admin_ui
from 智能体.agent.agent import agent
from 智能体.agent.report_service import ReportService
from 智能体.agent.sentiment_service import SentimentService, SentimentResult
from 智能体.config_handler import agent_conf
from 智能体.path_toll import get_abs_path

st.set_page_config(page_title="电商智能助手", layout="wide", initial_sidebar_state="expanded")

# 角色状态与左上角的角色切换控件必须先于页面内容渲染
admin_ui.init_state()
admin_ui.render_role_switcher()


@st.cache_resource
def get_report_service():
    return ReportService()


@st.cache_resource
def get_sentiment_service():
    return SentimentService()


def load_records_index():
    """从使用记录中取出可选的用户与月份"""
    path = get_abs_path(agent_conf["external_data_path"])
    users, months = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["用户ID"] not in users:
                users.append(row["用户ID"])
            if row["月份"] not in months:
                months.append(row["月份"])
    return sorted(users), sorted(months, reverse=True)


def ask_agent(question: str):
    """流式跑一遍 agent，返回最终回答与调用过的工具名"""
    answer, tools_used = "", []
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        message = chunk["messages"][-1]
        if getattr(message, "tool_calls", None):
            tools_used.extend(tc["name"] for tc in message.tool_calls)
        elif message.content:
            answer = message.content
    return answer, tools_used


def render_sentiment(result: SentimentResult):
    """把情感分析结果渲染成一行行标签"""
    st.markdown(
        f"- **情绪倾向**：{result.emotion}（强度 {result.emotion_level}/100）\n"
        f"- **紧急程度**：{result.urgency}\n"
        f"- **满意程度**：{result.satisfaction}\n"
        f"- **是否转接人工**：{'是' if result.need_human else '否'}（{result.need_human_reason}）\n"
        f"- **诉求摘要**：{result.summary}"
    )


def render_chat():
    # 输入内容分析仅管理员可见
    show_analysis = admin_ui.is_admin()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if show_analysis and message.get("sentiment"):
                with st.expander("输入内容分析"):
                    render_sentiment(message["sentiment"])

    question = st.chat_input("请输入问题，例如：扫地机器人无法回充怎么办？")
    if not question:
        return

    # 分析对所有人都照常计算并落库，只是渲染受角色控制
    with st.spinner("正在分析输入内容..."):
        sentiment = get_sentiment_service().analyze(question)
    admin_ui.record("user", question, sentiment.model_dump())

    st.session_state.messages.append(
        {"role": "user", "content": question, "sentiment": sentiment}
    )
    with st.chat_message("user"):
        st.markdown(question)
        if show_analysis:
            with st.expander("输入内容分析", expanded=True):
                render_sentiment(sentiment)

    if show_analysis and sentiment.need_human:
        st.warning(f"检测到该输入可能需要人工介入：{sentiment.need_human_reason}")

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并思考..."):
            answer, tools_used = ask_agent(question)
        if tools_used:
            st.caption("调用工具：" + "、".join(tools_used))
        st.markdown(answer)

    admin_ui.record("assistant", answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_report():
    users, months = load_records_index()

    col1, col2 = st.columns(2)
    user_id = col1.selectbox("用户ID", users)
    month = col2.selectbox("月份", months)

    if not st.button("生成使用情况报告", type="primary"):
        st.info("选择用户与月份后点击按钮，将基于使用记录与知识库生成报告。")
        return

    with st.spinner("正在查询使用记录并生成报告..."):
        report = get_report_service().generate_report(user_id, month)
    st.markdown(report)


st.title("电商智能助手")
# 「管理后台」只对已登录管理员动态生成
labels = ["智能问答", "使用报告"] + (["管理后台"] if admin_ui.is_admin() else [])
tabs = dict(zip(labels, st.tabs(labels)))

with tabs["智能问答"]:
    render_chat()

with tabs["使用报告"]:
    render_report()

if admin_ui.is_admin():
    with tabs["管理后台"]:
        admin_ui.render_admin_panel()
