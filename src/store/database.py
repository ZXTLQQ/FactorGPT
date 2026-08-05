"""SQLite 连接与表结构管理（src/store/database.py）。

设计目标
--------
1. 零依赖：仅用标准库 sqlite3，随项目落盘在 ``data/factorgpt.db``。
2. 线程安全：Streamlit 的脚本线程会在每次交互重建，因此不缓存长连接，
   而是「每次操作开一个连接」并配合进程内可重入锁，避免 ``check_same_thread`` 报错。
3. 幂等建表：``init_db()`` 可以被反复调用；新增列通过 ``_ensure_columns`` 平滑升级，
   老用户的数据库文件不会因为版本迭代而失效。
4. WAL 模式：读写并发更友好，避免界面刷新时被写锁阻塞。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# 项目根目录：src/store/database.py -> src/store -> src -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH: Path = Path(
    os.environ.get("FACTORGPT_DB", _PROJECT_ROOT / "data" / "factorgpt.db")
)

_LOCK = threading.RLock()
_INITIALIZED = False


# ---------------------------------------------------------------------------
# 路径管理
# ---------------------------------------------------------------------------
def get_db_path() -> Path:
    """返回当前数据库文件路径。"""
    return DB_PATH


def set_db_path(path: str | os.PathLike) -> None:
    """切换数据库文件（测试或多套工作区时使用），切换后自动重新建表。"""
    global DB_PATH, _INITIALIZED
    with _LOCK:
        DB_PATH = Path(path)
        _INITIALIZED = False
    init_db()


# ---------------------------------------------------------------------------
# 表结构
# ---------------------------------------------------------------------------
_SCHEMA: List[str] = [
    # —— 通用界面状态（操作记忆的核心：任意 key -> JSON） ——
    """
    CREATE TABLE IF NOT EXISTS app_state (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        scope       TEXT NOT NULL DEFAULT 'global',
        updated_at  TEXT NOT NULL
    )
    """,
    # —— 操作日志（时间线） ——
    """
    CREATE TABLE IF NOT EXISTS operation_log (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        ts       TEXT NOT NULL,
        module   TEXT NOT NULL,
        action   TEXT NOT NULL,
        summary  TEXT NOT NULL DEFAULT '',
        status   TEXT NOT NULL DEFAULT 'ok',
        payload  TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_oplog_ts ON operation_log(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_oplog_module ON operation_log(module)",
    # —— 因子体系 ——
    """
    CREATE TABLE IF NOT EXISTS factor_systems (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL UNIQUE,
        description  TEXT NOT NULL DEFAULT '',
        weight_mode  TEXT NOT NULL DEFAULT 'equal',
        config       TEXT NOT NULL DEFAULT '{}',
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_system_members (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        system_id     INTEGER NOT NULL,
        factor_name   TEXT NOT NULL,
        display_name  TEXT NOT NULL DEFAULT '',
        dimension     TEXT NOT NULL DEFAULT '未分类',
        category      TEXT NOT NULL DEFAULT '',
        source        TEXT NOT NULL DEFAULT 'static',
        direction     TEXT NOT NULL DEFAULT 'positive',
        weight        REAL NOT NULL DEFAULT 0.0,
        quality       REAL NOT NULL DEFAULT 0.5,
        code          TEXT NOT NULL DEFAULT '',
        meta          TEXT NOT NULL DEFAULT '{}',
        UNIQUE(system_id, factor_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_members_sys ON factor_system_members(system_id)",
    # —— 体系回测结果 ——
    """
    CREATE TABLE IF NOT EXISTS backtest_runs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        system_id    INTEGER,
        system_name  TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL,
        universe     TEXT NOT NULL DEFAULT '',
        period       TEXT NOT NULL DEFAULT '',
        params       TEXT NOT NULL DEFAULT '{}',
        metrics      TEXT NOT NULL DEFAULT '{}',
        details      TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_sys ON backtest_runs(system_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_ts ON backtest_runs(created_at DESC)",
    # —— 挖掘产出（Agent / GP / 精炼厂统一沉淀） ——
    """
    CREATE TABLE IF NOT EXISTS mining_records (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts           TEXT NOT NULL,
        module       TEXT NOT NULL DEFAULT 'agent',
        query        TEXT NOT NULL DEFAULT '',
        factor_name  TEXT NOT NULL DEFAULT '',
        expression   TEXT NOT NULL DEFAULT '',
        code         TEXT NOT NULL DEFAULT '',
        metrics      TEXT NOT NULL DEFAULT '{}',
        payload      TEXT NOT NULL DEFAULT '{}',
        starred      INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mining_ts ON mining_records(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mining_name ON mining_records(factor_name)",
    # —— Agent 对话历史 ——
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        title       TEXT NOT NULL DEFAULT '未命名会话',
        kind        TEXT NOT NULL DEFAULT 'agent',
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  INTEGER NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        ts          TEXT NOT NULL,
        meta        TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_msg_session ON chat_messages(session_id)",
]

# 版本迭代时的增量列：{表名: {列名: 列定义}}
_INCREMENTAL_COLUMNS: Dict[str, Dict[str, str]] = {
    "factor_systems": {"tags": "TEXT NOT NULL DEFAULT '[]'"},
    "mining_records": {"starred": "INTEGER NOT NULL DEFAULT 0"},
}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """为已存在的老库补齐新增列，保证平滑升级。"""
    for table, columns in _INCREMENTAL_COLUMNS.items():
        try:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.Error:
            continue
        if not existing:
            continue
        for col, ddl in columns.items():
            if col not in existing:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                except sqlite3.Error:
                    pass


# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """获取一个自动提交/回滚的数据库连接。"""
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db(force: bool = False) -> None:
    """建表（幂等）。首次调用时创建全部表结构并补齐增量列。"""
    global _INITIALIZED
    if _INITIALIZED and not force:
        return
    with _LOCK:
        if _INITIALIZED and not force:
            return
        conn = _connect()
        try:
            for stmt in _SCHEMA:
                conn.execute(stmt)
            _ensure_columns(conn)
            conn.commit()
            _INITIALIZED = True
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def now() -> str:
    """统一时间戳格式（本地时间，秒级）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dumps(obj: Any) -> str:
    """JSON 序列化，遇到不可序列化对象降级为字符串，绝不抛异常。"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"_repr": str(obj)}, ensure_ascii=False)


def loads(text: Optional[str], default: Any = None) -> Any:
    """JSON 反序列化，失败时返回默认值。"""
    if not text:
        return default if default is not None else {}
    try:
        return json.loads(text)
    except Exception:
        return default if default is not None else {}


def db_stats() -> Dict[str, int]:
    """各表行数统计，用于「操作记忆」页展示。"""
    tables = [
        "app_state",
        "operation_log",
        "factor_systems",
        "factor_system_members",
        "backtest_runs",
        "mining_records",
        "chat_sessions",
        "chat_messages",
    ]
    out: Dict[str, int] = {}
    with connection() as conn:
        for t in tables:
            try:
                out[t] = int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            except sqlite3.Error:
                out[t] = 0
    return out


def db_size_kb() -> float:
    """数据库文件体积（KB）。"""
    try:
        return round(DB_PATH.stat().st_size / 1024, 1)
    except OSError:
        return 0.0
