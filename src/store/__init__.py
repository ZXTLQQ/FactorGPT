"""FactorGPT 本地持久化层（SQLite）。

提供「操作记忆」能力：界面状态、操作日志、挖掘记录、因子体系、回测结果、
Agent 对话历史全部落盘到本地 SQLite，关闭应用后重新打开可完整恢复现场。

用法::

    from store import state, ops, systems, runs, mining, chats

    state.set("active_page", "因子体系搭建")
    ops.log("factor_system", "save", "保存体系: 多维Alpha体系")
"""

from .database import DB_PATH, connection, get_db_path, init_db, set_db_path
from .repository import (
    ChatRepository,
    MiningRepository,
    OperationRepository,
    RunRepository,
    StateRepository,
    SystemRepository,
    chats,
    mining,
    ops,
    runs,
    state,
    systems,
)

__all__ = [
    "DB_PATH",
    "connection",
    "get_db_path",
    "init_db",
    "set_db_path",
    "StateRepository",
    "OperationRepository",
    "SystemRepository",
    "RunRepository",
    "MiningRepository",
    "ChatRepository",
    "state",
    "ops",
    "systems",
    "runs",
    "mining",
    "chats",
]
