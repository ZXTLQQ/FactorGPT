import os, sys, traceback
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

lines = []
try:
    from data.market_data import MarketDataFetcher

    codes = ["600519", "000858", "000001"]
    q, err = MarketDataFetcher.quotes_for(codes)
    lines.append("QUOTES_OK=" + str(bool(q)) + " ERR=" + str(err))
    for c in codes:
        e = q.get(c, {})
        lines.append("  " + c + " " + str(e.get("名称")) + " " + str(e.get("最新价")) + " " + str(e.get("涨跌幅")))

    for code, nm in [("000001", "上证"), ("399006", "创业板"), ("899050", "北证50")]:
        d, derr = MarketDataFetcher.index_spot(code)
        lines.append("IDX " + nm + " " + str(None if d is None or d.empty else d.iloc[0].to_dict()) + " ERR=" + str(derr))

    k = MarketDataFetcher.stock_kline("600519", days=120, adjust="qfq")
    lines.append("STK_KLINE_ROWS=" + str(None if k is None else len(k)))
    ki = MarketDataFetcher.index_kline("000001", days=180)
    lines.append("IDX_KLINE_ROWS=" + str(None if ki is None else len(ki)))
    # 诊断主源返回
    import akshare as ak2
    try:
        em = ak2.stock_zh_a_hist(symbol="600519", period="daily", start_date="20260101", end_date="20260728", adjust="qfq")
        lines.append("EM_HIST_ROWS=" + str(None if em is None else len(em)))
    except Exception as e:
        lines.append("EM_HIST_ERR=" + repr(e)[:120])
    try:
        idh = __import__("data.index_query", fromlist=["IndexQueryService"]).IndexQueryService().get_index_hist("000001", start="20260101", end="20260728")
        lines.append("IDX_HIST_ROWS=" + str(None if idh is None else len(idh)))
    except Exception as e:
        lines.append("IDX_HIST_ERR=" + repr(e)[:120])

    rt = MarketDataFetcher.stock_realtime("600519")
    if isinstance(rt, tuple):
        rt = rt[0]
    lines.append("REALTIME=" + str(None if rt is None or getattr(rt, "empty", True) else rt.iloc[0].to_dict()))
except Exception:
    lines.append(traceback.format_exc())
finally:
    with open("verify_out.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
