"""失败模式库的回归测试（src/agent/failure_memory.py）。

要证明的是两件事：错误能被归到**正确的模式**，以及模式能变成 prompt 里真正
有用的负面约束——否则它只是又一个记日志的地方。
"""

import os
import sys
import types

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agent.failure_memory import FailureMemory, classify  # noqa: E402
from agent.nodes import FactorAgentNodes  # noqa: E402


def test_classify_known_patterns():
    cases = {
        "检测到前视偏差：factor 列与未来收益相关": "lookahead",
        "SVD did not converge in Linear least squares": "numerics",
        "divide by zero encountered in true_divide": "nan",
        "sandbox timeout after 20s": "timeout",
        "未找到 alpha_factor 函数": "no_entry",
        "因子退化成常数": "constant",
        "KeyError: 'turnover'": "missing_column",
    }
    for msg, kind in cases.items():
        assert classify(msg)[0] == kind, msg


def test_classify_unknown_falls_back():
    kind, hint = classify("some brand new error we have never seen")
    assert kind == "other" and hint


def test_record_accumulates(tmp_path):
    path = os.path.join(str(tmp_path), "fm.json")
    mem = FailureMemory(path)
    assert len(mem) == 0
    assert mem.prompt_block() == ""          # 没历史时不污染 prompt

    mem.record("SVD did not converge", "df['factor'] = 1")
    mem.record("divide by zero", "df['factor'] = 2")
    mem.record("SVD did not converge again")
    assert mem.cases["numerics"].count == 2
    assert mem.cases["nan"].count == 1
    assert mem.stats()["total"] == 3

    tops = mem.top(2)
    assert [c.kind for c in tops] == ["numerics", "nan"]
    assert os.path.exists(path)

    # 重新加载后累计次数不能丢
    assert FailureMemory(path).cases["numerics"].count == 2


def test_prompt_block_is_actionable(tmp_path):
    mem = FailureMemory(os.path.join(str(tmp_path), "fm.json"))
    for _ in range(3):
        mem.record("检测到前视偏差，请检查 shift")
    block = mem.prompt_block(n=2)
    assert "历史失败约束" in block
    assert "lookahead" in block and "shift(1)" in block
    assert "3 次" in block


def test_prompt_block_caps_and_sorts(tmp_path):
    mem = FailureMemory(os.path.join(str(tmp_path), "fm.json"))
    mem.record("timeout")
    for _ in range(5):
        mem.record("SVD did not converge")
    block = mem.prompt_block(n=1)
    assert "numerics" in block and "timeout" not in block


def test_nodes_injects_failure_block(tmp_path):
    """_failure_block 要把历史失败拼进生成提示；无历史时返回空串。"""
    empty = types.SimpleNamespace(failures=None)
    assert FactorAgentNodes._failure_block(empty) == ""

    bank = FailureMemory(os.path.join(str(tmp_path), "fm.json"))
    bank.record("SVD did not converge")
    obj = types.SimpleNamespace(failures=bank)
    block = FactorAgentNodes._failure_block(obj)
    assert "numerics" in block and block.startswith("\n")


def test_record_failure_silent_when_disabled():
    obj = types.SimpleNamespace(failures=None)
    assert FactorAgentNodes._record_failure(obj, "boom", "code") is None
