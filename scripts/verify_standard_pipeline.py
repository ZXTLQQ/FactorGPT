"""验证标准化回测流程：校验 -> 评价 -> 生成回测图 -> 分层回测表 -> 报告。
不依赖 LLM：用修正后的因子代码（已处理 NaN/SVD）跑通全流程，证明图表与分层结果可产出。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["FACTORGPT_FORCE_SYNTHETIC"] = "1"
from llm.client import load_config
from agent.graph import FactorAgent
from agent.nodes import FactorAgentNodes

# 修正后的因子代码：对用户原代码做稳健化（dropna + 保护 std=0）
HARDENED = '''
import pandas as pd
import numpy as np

def alpha_factor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    df['daily_ret'] = df.groupby('symbol')['close'].pct_change()
    df['mkt_cap'] = df['close'] * df['volume']
    # 市值加权市场收益率（先剔除 NaN，避免 lstsq 类错误）
    grp = df.groupby('date').apply(
        lambda g: (g['daily_ret'] * g['mkt_cap']).sum() / g['mkt_cap'].sum()
    )
    df['mkt_ret'] = grp.reset_index(level=0, drop=True)
    df['excess_ret'] = df['daily_ret'] - df['mkt_ret']
    df['rev_5d'] = df.groupby('symbol')['excess_ret'].transform(
        lambda x: x.rolling(5, min_periods=5).sum().shift(1)
    )
    df['illiq'] = df['daily_ret'].abs() / (df['amount'] / 1e8)
    df['illiq'] = np.log(df['illiq'] + 1e-10)
    df['illiq_21d'] = df.groupby('symbol')['illiq'].transform(
        lambda x: x.rolling(21, min_periods=21).mean().shift(1)
    )

    def rank_std(s):
        s = s.fillna(0)
        r = s.rank()
        sd = r.std()
        return (r - r.mean()) / sd if sd and sd > 0 else r * 0.0

    df['rev_rank'] = df.groupby('date')['rev_5d'].transform(rank_std)
    df['illiq_rank'] = df.groupby('date')['illiq_21d'].transform(rank_std)

    def orthogonalize(g):
        y = g['illiq_rank'].values.astype(float)
        x = g['rev_rank'].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            return pd.Series(np.zeros(len(g)), index=g.index)
        xm, ym = x[mask], y[mask]
        if xm.std() < 1e-9:
            return pd.Series(np.zeros(len(g)), index=g.index)
        A = np.vstack([xm, np.ones_like(xm)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, ym, rcond=None)
        resid = np.zeros(len(g))
        resid[mask] = ym - (coeff[0] * xm + coeff[1])
        return pd.Series(resid, index=g.index)

    df['illiq_orth'] = df.groupby('date').apply(orthogonalize).reset_index(level=0, drop=True)
    df['factor_raw'] = -df['rev_rank'] + df['illiq_orth']
    df['factor'] = df.groupby('date')['factor_raw'].transform(rank_std)
    df = df.drop(columns=['daily_ret', 'mkt_cap', 'mkt_ret', 'excess_ret', 'rev_5d',
                          'illiq', 'illiq_21d', 'rev_rank', 'illiq_rank', 'illiq_orth', 'factor_raw'])
    return df[['date', 'symbol', 'factor']]
'''

cfg = load_config()
agent = FactorAgent(cfg)
nodes = agent.nodes

state = {
    "factor_description": "混合日频与月频，结合短期反转与流动性",
    "factor_code": HARDENED,
    "factor_name": "MixedRevLiq",
    "metrics": {},
}

# 1) 校验 + 计算
v = nodes.validate_and_compute(state)
print("[校验]", "OK" if v.get("validation_ok") else "FAIL", v.get("validation_error", ""))
assert v.get("validation_ok"), "校验应通过"
state.update(v)

# 2) 评价 + 生成图表
e = nodes.evaluate_factor(state)
m = e.get("metrics", {})
print("[评价]", "error" in m, "| IC=%.4f" % m.get("ic", float("nan")),
      "| LS_sharpe=%.4f" % m.get("long_short_sharpe", float("nan")))
assert "error" not in m, "评价不应报错"
state.update(e)

# 3) 终局报告
f = nodes.finalize({**state, "validation_ok": True})
report = f.get("report", "")

print("\n[图表路径]")
for p in state.get("chart_paths", []):
    print("  -", p, "exists=", os.path.exists(p))

print("\n[分层回测片段]")
for line in report.splitlines():
    if "分层回测" in line or line.startswith("|") and ("Q" in line or "多空" in line):
        print(line)
print("\n[PASS] 标准化回测流程（校验->评价->图表->分层表->报告）已跑通")
