"""把旧的 SQLite 库（智能体/data/admin.db）一次性搬进 Docker 里的 PostgreSQL。

用法（在 d:\\PythonProject3 下）：
    .venv\\Scripts\\python.exe -m 智能体.admin.migrate_sqlite_to_pg

可重复执行：管理员按 username 去重（ON CONFLICT DO NOTHING），对话记录会重复插入，
所以正常只需在切换存储后跑一次。
"""

import sqlite3

from 智能体.admin import store
from 智能体.config_handler import admin_conf
from 智能体.path_toll import get_abs_path

# 旧 chat_log 的字段（不含新增的 tools_used）
_LEGACY_TURN_COLUMNS = store.COLUMNS[:-1]


def migrate() -> None:
    store.init_db()

    src = sqlite3.connect(get_abs_path(admin_conf["db_path"]))
    src.row_factory = sqlite3.Row
    try:
        admins = [dict(row) for row in src.execute(
            "SELECT username, salt, password_hash, created_at FROM admin_user"
        )]
        turns = [dict(row) for row in src.execute(
            f"SELECT {', '.join(_LEGACY_TURN_COLUMNS)} FROM chat_log ORDER BY id"
        )]
    finally:
        src.close()

    placeholders = ", ".join(["%s"] * len(store.COLUMNS))
    with store.connect() as conn:
        for row in admins:
            conn.execute(
                "INSERT INTO admin_user (username, salt, password_hash, created_at) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                (row["username"], row["salt"], row["password_hash"], row["created_at"]),
            )
        for row in turns:
            values = [row.get(col) for col in _LEGACY_TURN_COLUMNS] + [None]
            conn.execute(
                f"INSERT INTO chat_log ({', '.join(store.COLUMNS)}) VALUES ({placeholders})",
                values,
            )

    print(f"迁移管理员 {len(admins)} 个、对话记录 {len(turns)} 条")


if __name__ == "__main__":
    migrate()
    print(f"迁移后统计：{store.stats()}")
