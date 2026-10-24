import os

import yaml
from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo
from 智能体.path_toll import get_abs_path

# 加载项目根目录下的 .env，让密钥等敏感配置不写进代码
load_dotenv(get_abs_path(".env"), override=True)


def load_rag_config(config_path: str=get_abs_path("智能体/config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:# 打开配置文件
        return yaml.safe_load(f)# 加载配置文件


def load_chroma_config(config_path: str=get_abs_path("智能体/config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:# 打开配置文件
        return yaml.safe_load(f)# 加载配置文件


def load_prompt_config(config_path: str=get_abs_path("智能体/config/prompt.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_agent_config(config_path: str=get_abs_path("智能体/config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_admin_config(config_path: str=get_abs_path("智能体/config/admin.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_db_config(config_path: str=get_abs_path("智能体/config/db.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()

_required_chroma_keys = {
    "collection_name", "persist_directory", "k", "data_path",
    "md5_hex_store", "allow_knowledge_file_type",
    "chunk_size", "chunk_overlap", "separators",
}
_missing = _required_chroma_keys - set(chroma_conf or {})
if _missing:
    raise KeyError(f"chroma.yml 缺少配置项: {_missing}")

prompt_conf = load_prompt_config()
agent_conf = load_agent_config()

_required_agent_keys = {"external_data_path"}
_missing = _required_agent_keys - set(agent_conf or {})
if _missing:
    raise KeyError(f"agent.yml 缺少配置项: {_missing}")

admin_conf = load_admin_config()

_required_admin_keys = {"db_path", "invite_code"}
_missing = _required_admin_keys - set(admin_conf or {})
if _missing:
    raise KeyError(f"admin.yml 缺少配置项: {_missing}")

# 管理员邀请码优先从环境变量读取，仓库里的 admin.yml 不再存放真实邀请码
admin_conf["invite_code"] = (
    os.getenv("ADMIN_INVITE_CODE") or admin_conf.get("invite_code") or ""
)
if not admin_conf["invite_code"]:
    raise RuntimeError(
        "缺少 ADMIN_INVITE_CODE 环境变量，请在项目根目录的 .env 中配置管理员邀请码（可参考 .env.example）"
    )

db_conf = load_db_config()

_required_db_keys = {"dsn"}
_missing = _required_db_keys - set(db_conf or {})
if _missing:
    raise KeyError(f"db.yml 缺少配置项: {_missing}")

# 数据库连接串优先从环境变量读取，仓库里的 db.yml 不再存放真实连接串
db_conf["dsn"] = os.getenv("DATABASE_URL") or db_conf.get("dsn") or ""
if not db_conf["dsn"]:
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_name = os.getenv("POSTGRES_DB")
    if db_host and db_user and db_password and db_name:
        db_conf["dsn"] = make_conninfo(
            host=db_host, port=os.getenv("DB_PORT", "5432"),
            user=db_user, password=db_password, dbname=db_name,
        )
if not db_conf["dsn"]:
    raise RuntimeError(
        "缺少 DATABASE_URL 环境变量，请在项目根目录的 .env 中配置数据库连接串（可参考 .env.example）"
    )

if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
    print(chroma_conf)
    print(prompt_conf)
    print(agent_conf)
    print(admin_conf)
    print(db_conf)


