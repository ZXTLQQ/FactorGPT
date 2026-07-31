"""
行情后端数据库（短时缓存层）
================================

为行情中心提供轻量级、零依赖的本地持久化缓存：基于 SQLite 存储
实时报价、K 线、指数成分、新闻/研报等数据，显著降低对上游接口
（AKShare / 东方财富）的调用频率，支撑「自动刷新」下的短时复用。

设计要点：
- 单文件 SQLite（``data/cache.db``），无需安装额外依赖；
- 命名空间 + key + TTL 的通用 kv 存储，覆盖行情各类数据；
- 提供 ``get(key, ttl)``：命中且在有效期内的数据直接复用，否则回源；
- 支持手动失效（``invalidate``）与整体清理（``clear``）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

# 默认库位置：项目 data/ 目录下
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "cache.db")

# 命名空间常量（不同数据类型互不干扰）
NS_QUOTE = "quote"          # 实时报价（个股/指数）
NS_KLINE = "kline"          # K 线序列
NS_INTRADAY = "intraday"    # 当日分时
NS_CONSTITUENT = "cons"     # 指数成分股
NS_NEWS = "news"            # 个股新闻
NS_RESEARCH = "research"    # 个股研报
NS_INDEX_SPOT = "idx_spot"  # 指数实时点


class CacheDB:
    """线程安全的 SQLite 短缓存。"""

    _lock = threading.Lock()

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    ns   TEXT NOT NULL,
                    key  TEXT NOT NULL,
                    val  TEXT NOT NULL,
                    ts   REAL NOT NULL,
                    PRIMARY KEY (ns, key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_ts ON cache(ns, ts)"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _serialize(value: Any) -> str:
        # DataFrame 无法直 json，调用方应已转为 list[dict]
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def get(self, ns: str, key: str, ttl: int = 15) -> Optional[Any]:
        """读取缓存；命中且未过期返回解析后对象，否则返回 ``None``。"""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT val, ts FROM cache WHERE ns=? AND key=?", (ns, key)
            ).fetchone()
        if row is None:
            return None
        val, ts = row
        if (time.time() - ts) > ttl:
            return None
        try:
            return json.loads(val)
        except Exception:
            return val

    def set(self, ns: str, key: str, value: Any) -> None:
        """写入缓存（覆盖）。"""
        payload = self._serialize(value)
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO cache(ns, key, val, ts) VALUES(?,?,?,?) "
                "ON CONFLICT(ns, key) DO UPDATE SET val=excluded.val, ts=excluded.ts",
                (ns, key, payload, now),
            )

    def invalidate(self, ns: str, key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE ns=? AND key=?", (ns, key))

    def clear(self, ns: Optional[str] = None) -> int:
        """清理缓存；``ns=None`` 时清空全部。返回删除行数。"""
        with self._lock, self._connect() as conn:
            if ns is None:
                cur = conn.execute("DELETE FROM cache")
            else:
                cur = conn.execute("DELETE FROM cache WHERE ns=?", (ns,))
            return cur.rowcount

    def stats(self) -> dict:
        """返回各命名空间的缓存条数（用于「后端数据库」状态展示）。"""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ns, COUNT(*), MAX(ts) FROM cache GROUP BY ns"
            ).fetchall()
        return {ns: {"count": c, "last": mx} for ns, c, mx in rows}


# 模块级单例（避免重复建库）
_db = None


def get_cache_db() -> CacheDB:
    global _db
    if _db is None:
        _db = CacheDB()
    return _db
