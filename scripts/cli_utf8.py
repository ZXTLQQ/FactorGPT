"""CLI 脚本的第一步：把 stdout/stderr 强制成 UTF-8。

    from cli_utf8 import force_utf8_stdio
    force_utf8_stdio()

为什么必须有它：Windows 控制台的默认编码是 cp1252（或 GBK），** ``print`` 一句中文
就会抛 ``UnicodeEncodeError`` 直接结束进程**。在 CI 上这尤其阴险——构建其实已经成功，
但脚本最后的提示行崩了，于是整步被记为 failure：日志里最后一行是 Traceback，真正的
产出却已经落地，看上去像是"编译失败"。

这里不能依赖外部设置 ``PYTHONIOENCODING``：那是调用方的义务，脚本自己不该假设。
所以入口处自己 reconfigure；失败（被重定向到非 TextIO）就放过——它只影响提示语能否
显示，不该成为功能成败的条件。
"""
from __future__ import annotations

import sys


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover - 被重定向到非 TextIO
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 编码细节不该让脚本挂掉
            pass
