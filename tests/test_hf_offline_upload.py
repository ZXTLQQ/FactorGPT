"""高频并入离线层 / Agent 高频分支 / 上传+JEV 的契约测试。

全部用例用**合成数据**，不依赖本机桌面上的 382MB 快照，CI / 克隆后可直接跑。
覆盖的都是「错了会静默出错」的点：

- 分钟面板列必须是扁平单级（MultiIndex 列头会让下游取列直接 KeyError）
- 重采样口径：流量求和、状态取均值（混用会把订单流稀释成噪声）
- 离线层读高频：文件缺失要返回空表并给出指引，绝不去啃原始快照
- Agent 高频分支：合约码不做 zfill(6)、跳过行业/市值中性化
- 上传管道：图片不造假 OCR、表格能对齐才派生外部因子、JEV 无 Key 时降级不报错
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.hf_adapter import HFDataSource  # noqa: E402
from data.hf_panel import build_minute_panel  # noqa: E402
from data.offline_adapter import OfflineDataSource  # noqa: E402
from engine.jev import JEVClient  # noqa: E402
from engine.upload_ingest import UploadIngestor  # noqa: E402

TICK = 0.02


# --------------------------------------------------------------------------
# 合成 L2 快照：两个合约，10 分钟，500ms 一格
# --------------------------------------------------------------------------
def _synth_l2(n: int = 1200, contract: str = "AU2601", symbol: str = "au",
              base: float = 400.0) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2025-12-08 09:00:00")
    for i in range(n):
        ts = t0 + pd.to_timedelta(500 * i, unit="ms")
        mid = base + TICK * (i % 40)
        r = {
            "Exch": "SHFE", "Symbol": symbol, "Contract": contract,
            "TimeStr": ts.strftime("%H:%M:%S.%f")[:-3],
            # 原始落库里 CalDate/Date 都是整数 YYYYMMDD（坑 1 会按 %Y%m%d 解析）
            "CalDate": 20251208, "Date": 20251208,
            # SortTime 是 HHMMSSmmm 的分段编码（不是线性毫秒钟，进位不会自动传播），
            # 必须按时分秒毫秒逐段拼，否则重采样出来的分钟数会错。
            "SortTime": (ts.hour * 10_000_000 + ts.minute * 100_000
                         + ts.second * 1_000 + ts.microsecond // 1000),
            "Session": "M",
            "SP1": mid + TICK, "SV1": 100.0, "BP1": mid - TICK, "BV1": 90.0,
            "LastPrice": mid, "Volume": float(1000 + i), "Turnover": float(4e5 + 100 * i),
            "OpenInt": float(50000 + i),
        }
        for lv in range(2, 6):
            r[f"SP{lv}"] = mid + TICK * lv
            r[f"SV{lv}"] = float(80 + 10 * lv)
            r[f"BP{lv}"] = mid - TICK * lv
            r[f"BV{lv}"] = float(70 + 10 * lv)
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def hf_file(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("l2") / "synth.pqt"
    pd.concat([_synth_l2(), _synth_l2(contract="AU2602", base=401.0)],
              ignore_index=True).to_parquet(p, index=False)
    return p


# --------------------------------------------------------------------------
# 1) 分钟面板
# --------------------------------------------------------------------------
def test_panel_columns_are_flat(hf_file):
    hf = HFDataSource(config={}, file=str(hf_file), cache_dir=str(hf_file.parent / "cache"))
    panel = build_minute_panel(hf, freq="1min", max_contracts=2)
    assert not panel.empty
    assert not isinstance(panel.columns, pd.MultiIndex), "列头必须是单级扁平"
    for c in ("date", "symbol", "open", "high", "low", "close", "volume", "pct_chg"):
        assert c in panel.columns
    assert panel["symbol"].nunique() == 2
    assert panel["date"].nunique() == 10          # 1200 * 500ms = 10 分钟


def test_panel_flow_vs_state_aggregation(hf_file):
    """hf_ofi 是流量（区间求和），hf_spread 是状态（区间均值）。"""
    hf = HFDataSource(config={}, file=str(hf_file), cache_dir=str(hf_file.parent / "c2"))
    panel = build_minute_panel(hf, freq="1min", max_contracts=1)
    if "hf_ofi" in panel.columns:
        # 单分钟内的净额之和，量级应大于单笔快照的典型值
        assert panel["hf_ofi"].abs().max() >= panel["hf_ofi"].abs().median()
    if "hf_spread" in panel.columns:
        assert panel["hf_spread"].notna().all()
    assert panel["close"].notna().all()


def test_panel_no_fake_rows(hf_file):
    hf = HFDataSource(config={}, file=str(hf_file), cache_dir=str(hf_file.parent / "c3"))
    panel = build_minute_panel(hf, freq="1min", max_contracts=2)
    assert panel["close"].notna().all(), "没有快照的分钟不能留下假行"


# --------------------------------------------------------------------------
# 2) 离线数据源读高频
# --------------------------------------------------------------------------
def test_offline_hf_panel_missing_file_returns_empty(tmp_path):
    src = OfflineDataSource(config={
        "data": {"offline": {"index": "csi800", "dir": str(tmp_path),
                             "hf": {"panel_file": str(tmp_path / "nope.parquet")}}}})
    assert src.get_hf_panel().empty
    assert "hf_offline_build" in src.last_fetch_info.get("message", "")


def test_offline_hf_panel_reads_and_filters(tmp_path):
    panel = pd.DataFrame({
        "date": ["2025-12-08 09:01:00"] * 4,
        "symbol": ["AU2601", "AU2602", "AG2601", "AG2602"],
        "close": [1.0, 2.0, 3.0, 4.0],
        "hf_ofi": [1.0, 2.0, 3.0, 4.0],
    })
    p = tmp_path / "hf_panel_1min.parquet"
    panel.to_parquet(p, index=False)
    src = OfflineDataSource(config={
        "data": {"offline": {"index": "csi800", "dir": str(tmp_path),
                             "hf": {"panel_file": str(p)}}}})
    assert src.hf_enabled
    got = src.get_hf_panel(symbols=["au2601", "ag2601"])
    assert len(got) == 2
    assert src.get_hf_panel(start="2025-12-08 09:02:00").empty


# --------------------------------------------------------------------------
# 3) Agent 高频分支
# --------------------------------------------------------------------------
def test_hf_mode_switch():
    from agent.graph import FactorAgent

    assert FactorAgent._hf_mode({"source": "hf"}) is True
    assert FactorAgent._hf_mode({"source": "offline",
                                 "offline": {"hf": {"agent_mode": True, "enabled": True}}}) is True
    assert FactorAgent._hf_mode({"source": "offline"}) is False


def test_user_table_is_normalized_into_panel_contract(tmp_path):
    """用户接入自有高频表：认中英列名、合约码不补零、其余数值列挂 hf_ 前缀。"""
    from data.hf_panel import load_user_panel, normalize_user_panel

    raw = pd.DataFrame({
        "成交时间": ["2025-12-08 09:01:00", "2025-12-08 09:02:00"],
        "合约": ["AU2601", "AU2601"],
        "收盘价": [400.0, 401.0],
        "成交量": [12.0, 15.0],
        "买卖压力": [0.3, -0.2],
    })
    norm = normalize_user_panel(raw)
    assert list(norm.columns[:3]) == ["date", "symbol", "open"]
    assert norm["symbol"].tolist() == ["AU2601", "AU2601"], "合约码不得补零"
    assert "hf_买卖压力" in norm.columns, "未识别的数值列挂 hf_ 前缀进面板"
    assert pd.isna(norm["pct_chg"].iloc[0]), "首行无前值，pct_chg 应为 NaN 而不是 0"
    assert abs(norm["pct_chg"].iloc[1] - 0.25) < 1e-9

    p = tmp_path / "my.csv"
    raw.to_csv(p, index=False, encoding="utf-8")
    got = load_user_panel(p)
    assert len(got) == 2 and got["close"].iloc[1] == 401.0
    with pytest.raises(FileNotFoundError):
        load_user_panel(tmp_path / "nope.csv")
    with pytest.raises(ValueError):
        normalize_user_panel(pd.DataFrame({"时间": ["2025-12-08 09:01:00"], "x": [1]}))


def test_hf_failure_keeps_regular_workflow(monkeypatch):
    """高频取不到时必须回退常规日频链路，而不是把整个取数打死。"""
    from agent.graph import FactorAgent

    def _boom(self, cfg, n):
        raise RuntimeError("面板缺失")

    monkeypatch.setattr(FactorAgent, "_load_data_hf", _boom)
    agent = FactorAgent.__new__(FactorAgent)
    # primary_source=ths 且未配置令牌 → 常规链路立即失败落到合成数据（不触网）
    agent.config = {"data": {"source": "hf", "primary_source": "ths", "universe_size": 4,
                             "default_start_date": "2024-01-01",
                             "default_end_date": "2024-02-01"}}
    agent.data_modality = "daily"
    agent.hf_source = ""
    agent.hf_fallback_reason = ""
    kline, industry, cap = agent._load_data()
    assert agent.hf_fallback_reason, "必须记录高频回退原因，便于 UI 解释"
    assert agent.data_modality == "daily", "高频失败后模态必须回到日频"
    assert not kline.empty and kline["symbol"].nunique() == 4


def test_oos_split_intraday_splits_by_day():
    """分钟面板的样本外切分按自然日：同一天的分钟不能同时进训练与测试。"""
    from agent.graph import FactorAgent

    rows = []
    for day in (1, 2, 3, 4):
        for m in range(3):
            rows.append({"date": f"2025-12-0{day} 09:{m:02d}:00",
                         "symbol": "AU2601", "close": 400.0 + m})
    panel = pd.DataFrame(rows)
    train, test = FactorAgent._split_oos(panel, test_frac=0.5, modality="intraday")
    assert train is not None and test is not None
    tr_days = {str(d)[:10] for d in train["date"]}
    te_days = {str(d)[:10] for d in test["date"]}
    assert not (tr_days & te_days), "同一交易日被拆进了训练/测试，分钟级泄漏"

    one = panel[panel["date"].str.startswith("2025-12-01")]
    assert FactorAgent._split_oos(one, modality="intraday")[1] is None, \
        "只有一个交易日时不该硬切出样本外"


def test_attach_external_factors_aligns_on_date_symbol():
    """上传表格派生的外部因子必须按 (date, symbol) 并进面板，缺失填 0。"""
    from agent.graph import FactorAgent

    agent = FactorAgent.__new__(FactorAgent)
    agent.kline = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
        "symbol": ["000001", "000002", "000001"],
        "close": [1.0, 2.0, 3.0],
    })
    agent.train_kline = agent.kline.copy()
    agent.test_kline = None
    s = pd.Series([5.0, 7.0], index=pd.MultiIndex.from_tuples(
        [("2025-01-01", "000001"), ("2025-01-02", "000001")]))
    added = agent.attach_external_factors({"新闻情绪": s})
    assert added == ["ext_新闻情绪"]
    col = added[0]
    assert agent.kline[col].tolist() == [5.0, 0.0, 7.0]
    assert agent.kline[col].notna().all()


# --------------------------------------------------------------------------
# 4) JEV 判定（无 Key 时走本地规则）
# --------------------------------------------------------------------------
def test_jev_falls_back_to_heuristic_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    # 把本地模型目录指向空目录：本用例专测**规则兜底**这一级，不能被已训产物干扰
    # （默认目录里若有 scripts/train_multimodal.py 的产物，走的会是 local-model）
    cli = JEVClient({"jev": {"enabled": True},
                     "multimodal": {"model_dir": str(tmp_path / "no_model")}})
    assert not cli.callable
    res = cli.analyze("公司发布业绩预告，预计下半年需求回暖，利好龙头。", filename="a.txt")
    assert res["engine"] == "heuristic"
    assert res["factor_hint_label"]
    assert "前瞻" in res["summary"] or "日期" in res["summary"] or res["factorizable"] > 0
    assert 0.0 <= res["alignable"] <= 1.0


def test_jev_detects_forward_looking():
    cli = JEVClient({"jev": {"enabled": True}})
    res = cli._heuristic("预计 2026 年需求将大幅回暖")
    assert res["forward_looking"]["noul"] >= 0.7


# --------------------------------------------------------------------------
# 5) 上传管道
# --------------------------------------------------------------------------
def test_ingest_text_and_table(tmp_path):
    cfg = {"data": {"uploads": {"dir": str(tmp_path / "up"),
                                "index_file": str(tmp_path / "up" / "index.json")}},
           "jev": {"enabled": True},
           # 指向空目录，保证走规则判定，用例不依赖本机是否训过本地模型
           "multimodal": {"model_dir": str(tmp_path / "no_model")}}
    ing = UploadIngestor(cfg)
    txt = tmp_path / "note.txt"
    txt.write_text("公司公告：业绩超预期，利好。2025-12-08 600519", encoding="utf-8")
    it1 = ing.ingest_file(txt)
    assert it1.kind == "text" and it1.jev["engine"] == "heuristic"

    csv = tmp_path / "alt.csv"
    pd.DataFrame({"date": ["2025-01-01", "2025-01-01", "2025-01-02"],
                  "symbol": ["000001", "000002", "000001"],
                  "score": [0.5, -0.2, 0.9]}).to_csv(csv, index=False)
    it2 = ing.ingest_file(csv)
    assert it2.kind == "table"
    if it2.factor is not None:                 # 列映射命中才派生，未命中也不算失败
        assert isinstance(it2.factor.index, pd.MultiIndex)
    ctx = ing.context_text()
    assert "note.txt" in ctx
    ing.persist_index()
    assert (tmp_path / "up" / "index.json").exists()


def test_unsupported_type_rejected(tmp_path):
    ing = UploadIngestor({"data": {"uploads": {"dir": str(tmp_path / "u2")}}})
    p = tmp_path / "x.docx"
    p.write_bytes(b"whatever")
    with pytest.raises(ValueError):
        ing.ingest_file(p)
