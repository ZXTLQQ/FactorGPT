"""因子沙箱子进程运行脚本（由 FactorSandbox._run_subprocess 调用）。

仅负责：在独立进程中执行因子代码并返回原始结果（pickle），
由父进程统一做 _normalize。沿用主进程的受限 globals，防止逃逸。
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factor_builder import _SAFE_BUILTINS  # noqa: E402


def main() -> None:
    code_path, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
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

    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
