"""领域仓储层（src/store/repository.py）。

把裸 SQL 收敛在这里，界面层只面对 Python 字典 / 列表。
每个仓储都是无状态的，模块底部提供开箱即用的单例。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .database import connection, dumps, loads, now


# ===========================================================================
# 1. 界面状态（操作记忆核心）
# ===========================================================================
class StateRepository:
    """任意 key -> JSON 值的持久化字典，用于记住用户的界面选择。"""

    def set(self, key: str, value: Any, scope: str = "global") -> None:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO app_state(key, value, scope, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    scope = excluded.scope,
                    updated_at = excluded.updated_at
                """,
                (key, dumps(value), scope, now()),
            )

    def get(self, key: str, default: Any = None) -> Any:
        with connection() as conn:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        val = loads(row["value"], default={"__missing__": True})
        if isinstance(val, dict) and val.get("__missing__"):
            return default
        return val

    def set_many(self, mapping: Dict[str, Any], scope: str = "global") -> None:
        ts = now()
        with connection() as conn:
            conn.executemany(
                """
                INSERT INTO app_state(key, value, scope, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value, updated_at = excluded.updated_at
                """,
                [(k, dumps(v), scope, ts) for k, v in mapping.items()],
            )

    def all(self, scope: Optional[str] = None) -> Dict[str, Any]:
        sql = "SELECT key, value FROM app_state"
        args: tuple = ()
        if scope:
            sql += " WHERE scope = ?"
            args = (scope,)
        with connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        return {r["key"]: loads(r["value"]) for r in rows}

    def delete(self, key: str) -> None:
        with connection() as conn:
            conn.execute("DELETE FROM app_state WHERE key = ?", (key,))

    def clear(self, scope: Optional[str] = None) -> None:
        with connection() as conn:
            if scope:
                conn.execute("DELETE FROM app_state WHERE scope = ?", (scope,))
            else:
                conn.execute("DELETE FROM app_state")


# ===========================================================================
# 2. 操作日志
# ===========================================================================
class OperationRepository:
    """记录用户的关键动作，形成可回溯的操作时间线。"""

    def log(
        self,
        module: str,
        action: str,
        summary: str = "",
        status: str = "ok",
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        with connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO operation_log(ts, module, action, summary, status, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now(), module, action, summary, status, dumps(payload or {})),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 100, module: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM operation_log"
        args: List[Any] = []
        if module:
            sql += " WHERE module = ?"
            args.append(module)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "module": r["module"],
                "action": r["action"],
                "summary": r["summary"],
                "status": r["status"],
                "payload": loads(r["payload"]),
            }
            for r in rows
        ]

    def module_counts(self) -> Dict[str, int]:
        with connection() as conn:
            rows = conn.execute(
                "SELECT module, COUNT(*) AS n FROM operation_log GROUP BY module ORDER BY n DESC"
            ).fetchall()
        return {r["module"]: int(r["n"]) for r in rows}

    def clear(self) -> None:
        with connection() as conn:
            conn.execute("DELETE FROM operation_log")


# ===========================================================================
# 3. 因子体系
# ===========================================================================
class SystemRepository:
    """因子体系（体系元信息 + 成分因子）的增删改查。"""

    def save(
        self,
        name: str,
        members: List[Dict[str, Any]],
        description: str = "",
        weight_mode: str = "equal",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """新建或整体覆盖一个因子体系，返回体系 id。"""
        ts = now()
        with connection() as conn:
            row = conn.execute(
                "SELECT id, created_at FROM factor_systems WHERE name = ?", (name,)
            ).fetchone()
            if row:
                sid = int(row["id"])
                conn.execute(
                    """
                    UPDATE factor_systems
                       SET description = ?, weight_mode = ?, config = ?, tags = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (description, weight_mode, dumps(config or {}), dumps(tags or []), ts, sid),
                )
                conn.execute("DELETE FROM factor_system_members WHERE system_id = ?", (sid,))
            else:
                cur = conn.execute(
                    """
                    INSERT INTO factor_systems
                        (name, description, weight_mode, config, tags, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, description, weight_mode, dumps(config or {}), dumps(tags or []), ts, ts),
                )
                sid = int(cur.lastrowid)

            conn.executemany(
                """
                INSERT INTO factor_system_members
                    (system_id, factor_name, display_name, dimension, category,
                     source, direction, weight, quality, code, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        sid,
                        m.get("factor_name") or m.get("name", ""),
                        m.get("display_name", ""),
                        m.get("dimension", "未分类"),
                        m.get("category", ""),
                        m.get("source", "static"),
                        m.get("direction", "positive"),
                        float(m.get("weight", 0.0) or 0.0),
                        float(m.get("quality", 0.5) or 0.0),
                        m.get("code", ""),
                        dumps(m.get("meta", {})),
                    )
                    for m in members
                ],
            )
        return sid

    def list(self) -> List[Dict[str, Any]]:
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT s.*, (SELECT COUNT(*) FROM factor_system_members m
                             WHERE m.system_id = s.id) AS n_factors
                  FROM factor_systems s
                 ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "weight_mode": r["weight_mode"],
                "config": loads(r["config"]),
                "tags": loads(r["tags"], default=[]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "n_factors": int(r["n_factors"]),
            }
            for r in rows
        ]

    def get(self, system_id: int) -> Optional[Dict[str, Any]]:
        with connection() as conn:
            r = conn.execute(
                "SELECT * FROM factor_systems WHERE id = ?", (system_id,)
            ).fetchone()
            if r is None:
                return None
            members = conn.execute(
                "SELECT * FROM factor_system_members WHERE system_id = ? ORDER BY id",
                (system_id,),
            ).fetchall()
        return {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "weight_mode": r["weight_mode"],
            "config": loads(r["config"]),
            "tags": loads(r["tags"], default=[]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "members": [
                {
                    "factor_name": m["factor_name"],
                    "display_name": m["display_name"],
                    "dimension": m["dimension"],
                    "category": m["category"],
                    "source": m["source"],
                    "direction": m["direction"],
                    "weight": float(m["weight"]),
                    "quality": float(m["quality"]),
                    "code": m["code"],
                    "meta": loads(m["meta"]),
                }
                for m in members
            ],
        }

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with connection() as conn:
            r = conn.execute(
                "SELECT id FROM factor_systems WHERE name = ?", (name,)
            ).fetchone()
        return self.get(int(r["id"])) if r else None

    def delete(self, system_id: int) -> None:
        with connection() as conn:
            conn.execute("DELETE FROM factor_system_members WHERE system_id = ?", (system_id,))
            conn.execute("DELETE FROM factor_systems WHERE id = ?", (system_id,))

    def rename(self, system_id: int, new_name: str) -> None:
        with connection() as conn:
            conn.execute(
                "UPDATE factor_systems SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, now(), system_id),
            )


# ===========================================================================
# 4. 体系回测结果
# ===========================================================================
class RunRepository:
    """体系回测运行记录，支持横向对比历史表现。"""

    def save(
        self,
        system_id: Optional[int],
        system_name: str,
        metrics: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        universe: str = "",
        period: str = "",
    ) -> int:
        with connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_runs
                    (system_id, system_name, created_at, universe, period, params, metrics, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    system_id,
                    system_name,
                    now(),
                    universe,
                    period,
                    dumps(params or {}),
                    dumps(metrics or {}),
                    dumps(details or {}),
                ),
            )
            return int(cur.lastrowid)

    def list(self, system_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM backtest_runs"
        args: List[Any] = []
        if system_id is not None:
            sql += " WHERE system_id = ?"
            args.append(system_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row(r) for r in rows]

    def get(self, run_id: int) -> Optional[Dict[str, Any]]:
        with connection() as conn:
            r = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()
        return self._row(r) if r else None

    def latest(self, system_id: int) -> Optional[Dict[str, Any]]:
        rows = self.list(system_id=system_id, limit=1)
        return rows[0] if rows else None

    def delete(self, run_id: int) -> None:
        with connection() as conn:
            conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))

    @staticmethod
    def _row(r) -> Dict[str, Any]:
        return {
            "id": r["id"],
            "system_id": r["system_id"],
            "system_name": r["system_name"],
            "created_at": r["created_at"],
            "universe": r["universe"],
            "period": r["period"],
            "params": loads(r["params"]),
            "metrics": loads(r["metrics"]),
            "details": loads(r["details"]),
        }


# ===========================================================================
# 5. 挖掘产出
# ===========================================================================
class MiningRepository:
    """Agent / GP / 精炼厂产出的因子统一沉淀，供因子体系搭建时挑选。"""

    def add(
        self,
        factor_name: str,
        module: str = "agent",
        query: str = "",
        expression: str = "",
        code: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        with connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO mining_records
                    (ts, module, query, factor_name, expression, code, metrics, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now(),
                    module,
                    query,
                    factor_name,
                    expression,
                    code,
                    dumps(metrics or {}),
                    dumps(payload or {}),
                ),
            )
            return int(cur.lastrowid)

    def add_many(self, records: List[Dict[str, Any]], module: str = "agent") -> int:
        if not records:
            return 0
        ts = now()
        with connection() as conn:
            conn.executemany(
                """
                INSERT INTO mining_records
                    (ts, module, query, factor_name, expression, code, metrics, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ts,
                        rec.get("module", module),
                        rec.get("query", ""),
                        rec.get("factor_name", ""),
                        rec.get("expression", ""),
                        rec.get("code", ""),
                        dumps(rec.get("metrics", {})),
                        dumps(rec.get("payload", {})),
                    )
                    for rec in records
                ],
            )
        return len(records)

    def list(
        self,
        limit: int = 200,
        module: Optional[str] = None,
        starred_only: bool = False,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM mining_records"
        clauses: List[str] = []
        args: List[Any] = []
        if module:
            clauses.append("module = ?")
            args.append(module)
        if starred_only:
            clauses.append("starred = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "module": r["module"],
                "query": r["query"],
                "factor_name": r["factor_name"],
                "expression": r["expression"],
                "code": r["code"],
                "metrics": loads(r["metrics"]),
                "payload": loads(r["payload"]),
                "starred": bool(r["starred"]),
            }
            for r in rows
        ]

    def toggle_star(self, record_id: int) -> None:
        with connection() as conn:
            conn.execute(
                "UPDATE mining_records SET starred = 1 - starred WHERE id = ?", (record_id,)
            )

    def delete(self, record_id: int) -> None:
        with connection() as conn:
            conn.execute("DELETE FROM mining_records WHERE id = ?", (record_id,))

    def clear(self) -> None:
        with connection() as conn:
            conn.execute("DELETE FROM mining_records")


# ===========================================================================
# 6. 对话历史
# ===========================================================================
class ChatRepository:
    """Agent 多轮对话的落盘，重开应用后可继续上一次的讨论。"""

    def ensure_session(self, title: str = "未命名会话", kind: str = "agent") -> int:
        ts = now()
        with connection() as conn:
            r = conn.execute(
                "SELECT id FROM chat_sessions WHERE kind = ? ORDER BY updated_at DESC LIMIT 1",
                (kind,),
            ).fetchone()
            if r:
                return int(r["id"])
            cur = conn.execute(
                "INSERT INTO chat_sessions(title, kind, created_at, updated_at) VALUES (?,?,?,?)",
                (title, kind, ts, ts),
            )
            return int(cur.lastrowid)

    def new_session(self, title: str = "未命名会话", kind: str = "agent") -> int:
        ts = now()
        with connection() as conn:
            cur = conn.execute(
                "INSERT INTO chat_sessions(title, kind, created_at, updated_at) VALUES (?,?,?,?)",
                (title, kind, ts, ts),
            )
            return int(cur.lastrowid)

    def list_sessions(self, kind: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql = (
            "SELECT s.*, (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS n "
            "FROM chat_sessions s"
        )
        args: List[Any] = []
        if kind:
            sql += " WHERE s.kind = ?"
            args.append(kind)
        sql += " ORDER BY s.updated_at DESC LIMIT ?"
        args.append(limit)
        with connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "kind": r["kind"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "n_messages": int(r["n"]),
            }
            for r in rows
        ]

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        ts = now()
        with connection() as conn:
            cur = conn.execute(
                "INSERT INTO chat_messages(session_id, role, content, ts, meta) VALUES (?,?,?,?,?)",
                (session_id, role, content, ts, dumps(meta or {})),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (ts, session_id)
            )
            return int(cur.lastrowid)

    def messages(self, session_id: int) -> List[Dict[str, Any]]:
        with connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "ts": r["ts"],
                "meta": loads(r["meta"]),
            }
            for r in rows
        ]

    def rename(self, session_id: int, title: str) -> None:
        with connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now(), session_id),
            )

    def delete_session(self, session_id: int) -> None:
        with connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
state = StateRepository()
ops = OperationRepository()
systems = SystemRepository()
runs = RunRepository()
mining = MiningRepository()
chats = ChatRepository()
