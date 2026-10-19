import hashlib
import os
from datetime import datetime

import psycopg

from 智能体.admin.store import connect, init_db
from 智能体.config_handler import admin_conf

_ITERATIONS = 200_000
_SALT_BYTES = 16
# 用户不存在时也做一次哈希，避免通过响应耗时差异枚举出有哪些用户名
_DUMMY_SALT = b"\x00" * _SALT_BYTES


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS).hex()


def register(username: str, password: str, invite_code: str) -> tuple[bool, str]:
    """注册管理员账号，需要匹配 admin.yml 里的邀请码"""
    username = (username or "").strip()
    if not username:
        return False, "用户名不能为空"
    if len(password or "") < 6:
        return False, "密码至少 6 位"
    if invite_code != admin_conf["invite_code"]:
        return False, "邀请码不正确"

    salt = os.urandom(_SALT_BYTES)
    # try 必须包住整个 with：PostgreSQL 报错后事务已中止，在事务内继续查询会抛 InFailedSqlTransaction
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO admin_user (username, salt, password_hash, created_at) "
                "VALUES (%s, %s, %s, %s)",
                (
                    username,
                    salt.hex(),
                    _hash(password, salt),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
    except psycopg.errors.UniqueViolation:
        return False, f"用户名 {username} 已被注册"
    return True, "注册成功"


def authenticate(username: str, password: str) -> bool:
    """校验管理员用户名与密码"""
    with connect() as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM admin_user WHERE username = %s",
            ((username or "").strip(),),
        ).fetchone()

    if row is None:
        _hash(password or "", _DUMMY_SALT)
        return False
    return _hash(password or "", bytes.fromhex(row["salt"])) == row["password_hash"]


if __name__ == "__main__":
    init_db()
    name = "selftest_admin"
    print("注册：", register(name, "demo123456", admin_conf["invite_code"]))
    print("正确密码：", authenticate(name, "demo123456"))
    print("错误密码：", authenticate(name, "wrong-password"))
    print("不存在的用户：", authenticate("no_such_admin", "demo123456"))
    print("错误邀请码注册：", register("selftest_admin2", "demo123456", "wrong-code"))

    conn = connect()
    try:
        conn.execute("DELETE FROM admin_user WHERE username LIKE 'selftest_admin%'")
        conn.commit()
    finally:
        conn.close()
    print("自测账号已清理")
