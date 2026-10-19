import uuid

import streamlit as st

from 智能体.admin import auth, store
from 智能体.admin.store import DISPLAY_COLUMNS
from 智能体.logger_hander import logger

ROLES = ["普通用户", "管理员"]


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
    st.session_state.setdefault("admin_user", None)
    st.session_state.setdefault("role", ROLES[0])
    store.init_db()


def is_admin() -> bool:
    return bool(st.session_state.get("admin_user"))


def strip_sentiment() -> None:
    """切回普通用户时把历史消息里的分析结果就地删掉，避免再切回管理员时读到残留"""
    if is_admin():
        return
    for message in st.session_state.get("messages", []):
        message.pop("sentiment", None)


def record(role: str, content: str, sentiment=None) -> None:
    """把一轮对话写入审计表，写库失败不影响对话"""
    try:
        store.log_turn(st.session_state["session_id"], role, content, sentiment)
    except Exception as e:
        logger.error(f"[record]写入对话记录失败，{str(e)}")


def render_role_switcher() -> None:
    """页面左上角（侧边栏顶部）的角色切换与管理员的登录/注册入口"""
    # 上一轮的「退出登录」在这里先把角色切回普通用户，必须早于下面的控件创建
    if st.session_state.pop("_switch_to_user", False):
        st.session_state["role"] = ROLES[0]
        st.session_state["admin_user"] = None

    with st.sidebar:
        st.markdown("### 角色切换")
        st.segmented_control("当前角色", ROLES, key="role", label_visibility="collapsed")

        if st.session_state["role"] == "管理员":
            if is_admin():
                st.success(f"已登录管理员：{st.session_state['admin_user']}")
                if st.button("退出登录"):
                    st.session_state["_switch_to_user"] = True
                    st.rerun()
            else:
                _render_login_forms()
        else:
            st.caption("普通用户无需登录，可直接问答")

        st.divider()
        st.caption(f"会话 ID：{st.session_state['session_id']}")

    strip_sentiment()


def _render_login_forms() -> None:
    tab_login, tab_register = st.tabs(["登录", "注册"])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            if st.form_submit_button("登录", type="primary"):
                if auth.authenticate(username, password):
                    st.session_state["admin_user"] = username.strip()
                    st.rerun()
                else:
                    st.error("用户名或密码错误")
    with tab_register:
        with st.form("register_form"):
            username = st.text_input("用户名", key="register_username")
            password = st.text_input("密码（至少 6 位）", type="password", key="register_password")
            invite_code = st.text_input("管理员邀请码", type="password", key="register_invite")
            if st.form_submit_button("注册", type="primary"):
                ok, message = auth.register(username, password, invite_code)
                if ok:
                    st.session_state["admin_user"] = username.strip()
                    st.rerun()
                else:
                    st.error(message)


def render_admin_panel() -> None:
    """管理后台：统计 + 全部/按会话分组查看 + 筛选 + CSV 导出 + 清理历史记录"""
    flash = st.session_state.pop("admin_flash", None)
    if flash:
        st.success(flash)

    data = store.stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("用户输入条数", data["total"])
    col2.metric("会话数", data["sessions"])
    col3.metric("需转接人工", data["need_human"])
    col4.metric("高紧急", data["urgent"])

    if data["emotions"]:
        st.caption("情绪倾向分布：" + "、".join(f"{k} {v}" for k, v in data["emotions"].items()))

    st.divider()
    sessions = store.list_sessions()
    if not sessions:
        st.info("暂无对话记录。普通用户在「智能问答」里发送消息后，这里就会出现记录。")
        return

    st.session_state.setdefault("admin_view_mode", "全部记录")
    view_mode = st.segmented_control(
        "查看方式", ["全部记录", "按会话分组"], key="admin_view_mode"
    ) or "全部记录"

    session_id = None
    if view_mode == "按会话分组":
        session_choices = {
            f"{s['session_id']} · {s['total']} 条 · 最后 {s['last_time']}": s["session_id"]
            for s in sessions
        }
        session_id = session_choices[st.selectbox("选择会话", list(session_choices), key="admin_session")]
        current = next(s for s in sessions if s["session_id"] == session_id)
        st.caption(
            f"该会话共 {current['total']} 条记录（其中用户输入 {current['user_count']} 条、"
            f"需转接人工 {current['need_human_count']} 条），首次 {current['first_time']}"
        )

    col1, col2, col3 = st.columns([2, 1, 1])
    keyword = col1.text_input("关键词搜索（匹配输入原文）", key="admin_keyword")
    urgency = col2.selectbox("紧急程度", ["全部", "低", "中", "高"], key="admin_urgency")
    only_human = col3.checkbox("仅看需转接人工", key="admin_only_human")

    rows = store.query_turns(
        session_id=session_id,
        keyword=keyword.strip() or None,
        urgency=None if urgency == "全部" else urgency,
        need_human=True if only_human else None,
    )

    st.caption(f"共 {len(rows)} 条记录（按时间倒序，最多显示 500 条）")
    st.dataframe(
        [
            {
                DISPLAY_COLUMNS[col]: (
                    ("是" if row[col] else "否")
                    if col == "need_human" and row[col] is not None
                    else ("" if row[col] is None else row[col])
                )
                for col in DISPLAY_COLUMNS
            }
            for row in rows
        ],
        width="stretch",
    )
    st.download_button(
        "下载为 CSV",
        data=store.to_csv(rows),
        file_name="chat_log.csv",
        mime="text/csv",
    )

    _render_cleanup(sessions)


def _render_cleanup(sessions: list[dict]) -> None:
    """清理历史记录：可只清某个会话，也可清空全部"""
    with st.expander("清理历史记录"):
        session_choices = {
            f"{s['session_id']} · {s['total']} 条 · 最后 {s['last_time']}": s["session_id"]
            for s in sessions
        }
        target = st.selectbox(
            "清理范围", ["清空全部记录"] + list(session_choices), key="admin_purge_target"
        )
        confirmed = st.checkbox("我确认删除，删除后不可恢复", key="admin_purge_confirm")

        if st.button("执行清理", type="primary", disabled=not confirmed):
            deleted = store.delete_turns(
                None if target == "清空全部记录" else session_choices[target]
            )
            st.session_state["admin_flash"] = f"已删除 {deleted} 条记录"
            st.rerun()
