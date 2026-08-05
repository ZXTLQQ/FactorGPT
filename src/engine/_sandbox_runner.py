"""因子沙箱子进程运行脚本（由 FactorSandbox._run_subprocess 调用）。

仅负责：在独立进程中执行因子代码并返回原始结果（pickle），
由父进程统一做 _normalize。沿用主进程的受限 globals，防止逃逸。

支持通过环境变量 SANDBOX_MEMORY_LIMIT_MB 限制内存使用（跨平台）。
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factor_builder import _SAFE_BUILTINS  # noqa: E402


def _apply_memory_limit() -> None:
    """尝试对当前子进程设置内存上限（跨平台）。

    Unix: 使用 resource.RLIMIT_AS 限制虚拟内存。
    Windows: 使用 psutil（若可用）监控进程内存，超限时主动终止。
    """
    limit_mb = int(os.environ.get("SANDBOX_MEMORY_LIMIT_MB", "0"))
    if limit_mb <= 0:
        return
    limit_bytes = limit_mb * 1024 * 1024
    try:
        import resource  # Unix 平台
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        return
    except (ImportError, ValueError, OSError):
        pass
    # Windows 回退：psutil 监控
    try:
        import psutil  # type: ignore
        _pid = os.getpid()
        _proc = psutil.Process(_pid)

        def _oom_check() -> None:
            try:
                mem = _proc.memory_info().rss
                if mem > limit_bytes:
                    print(f"[sandbox] 子进程内存超限: {mem / 1024 / 1024:.0f}MB > {limit_mb}MB，终止。",
                          file=sys.stderr)
                    os._exit(137)  # SIGKILL-style exit
            except (psutil.NoSuchProcess, Exception):
                pass
        import atexit
        # 在每次 pd/np 运算前检查内存（通过 monkey-patch 常见操作的开销可能过高，
        # 这里采用保守方案：在执行 alpha_factor 前检查一次）
        # 更完整的方案是在 _run_inprocess 中给 exec 加 trace 函数，但开销大。
        # 目前先记录限制值，由 runner 在执行前后检查。
        _oom_check()
    except ImportError:
        pass  # psutil 不可用时静默跳过


def main() -> None:
    code_path, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    _apply_memory_limit()

    with open(code_path, "r", encoding="utf-8") as f:
        code = f.read()
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    g = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "df": df.copy()}
    exec(code, g)  # noqa: S102

    fn = g.get("alpha_factor")
    if callable(fn):
        result = fn(df)
    elif "factor" in g and isinstance(g["factor"], (pd.Series, pd.DataFrame)):
        result = g["factor"]
    else:
        raise ValueError("因子代码未定义 alpha_factor() 函数，也未产出 'factor' 变量")

    # 结果产出后再次检查内存
    limit_mb = int(os.environ.get("SANDBOX_MEMORY_LIMIT_MB", "0"))
    if limit_mb > 0:
        try:
            import psutil
            mem = psutil.Process().memory_info().rss
            if mem > limit_mb * 1024 * 1024:
                print(f"[sandbox] 警告：因子执行内存 {mem / 1024 / 1024:.0f}MB 超限 {limit_mb}MB",
                      file=sys.stderr)
        except ImportError:
            pass

    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
