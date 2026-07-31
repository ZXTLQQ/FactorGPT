import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import pandas as pd
from data.market_data import _normalize_kline

# 英文列（新浪/Tushare）
df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "open": [10.0, 10.1],
                   "high": [10.5, 10.6], "low": [9.8, 9.9], "close": [10.2, 10.3],
                   "volume": [1000, 1100]})
r1 = _normalize_kline(df)
line1 = "ENG->CN: " + str(list(r1.columns))

# 中文列（东财/同花顺），保持不变
df2 = pd.DataFrame({"日期": ["2024-01-01"], "开盘": [1.0], "收盘": [2.0],
                    "最高": [2.1], "最低": [0.9], "成交量": [500]})
line2 = "CN keep: " + str(list(_normalize_kline(df2).columns))

# 空/None
line3 = "None-> " + str(_normalize_kline(None))
line4 = "empty-> " + str(list(_normalize_kline(pd.DataFrame()).columns))

assert "收盘" in r1.columns and "日期" in r1.columns and "成交量" in r1.columns
ok = "ALL_OK"
with open("norm_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join([line1, line2, line3, line4, ok]))
