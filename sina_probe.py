import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
from src.data.market_data import MarketDataFetcher

# 1) 成分股行情（用上证50成分股较少，但用沪深300太多；用上证指数成分太大，改测 quotes_for 几只）
codes = ["600519", "000858", "000001"]
q, err = MarketDataFetcher.quotes_for(codes)
print("QUOTES_OK=", bool(q), "ERR=", err)
for c in codes:
    print(" ", c, q.get(c, {}).get("名称"), q.get(c, {}).get("最新价"), q.get(c, {}).get("涨跌幅"))

# 2) 指数实时
for code, nm in [("000001", "上证"), ("399006", "创业板"), ("899050", "北证50")]:
    d = MarketDataFetcher.index_spot(code)
    print("IDX", nm, None if d is None or d.empty else d.iloc[0].to_dict())

# 3) K 线（股票 + 指数）
k = MarketDataFetcher.stock_kline("600519", days=120, adjust="qfq")
print("STK_KLINE_ROWS=", None if k is None else len(k))
ki = MarketDataFetcher.index_kline("000001", days=180)
print("IDX_KLINE_ROWS=", None if ki is None else len(ki))

# 4) 个股实时弹窗
rt = MarketDataFetcher.stock_realtime("600519")
print("REALTIME=", None if rt is None or rt.empty else rt.iloc[0].to_dict())
