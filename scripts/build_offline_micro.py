"""离线微观数据补充（行业 / 板块 / 市值 / 估值 / 地区）。

数据源（构建期联网、运行期完全离线）::

    1. 东财公司概况报表 ``RPT_F10_BASIC_ORGINFO``（datacenter-web 主机，全表分页，
       5000 条/页共 5 次请求）：给出名称、东财三级行业 ``EM2016``（一级/二级/三级）、
       证监会行业、注册地省份——覆盖全部 A 股（含科创板/创业板/北交所）。
    2. 腾讯行情 ``qt.gtimg.cn`` 批量报价（50 只/次，约 40 次请求）：最新价、总市值、
       流通市值、PE(TTM)、PB。
    3. 市场板块（沪市主板/深市主板/创业板/科创板/北交所）由代码前缀本地判定，不需联网。
    4. 新浪行业板块 ``newSinaHy.php`` 仅作为①不可用时的行业兜底通道。

产出 ``data/offline/micro_snapshot.parquet``，供 ``OfflineDataSource`` 的
``get_industry_and_cap`` / ``get_industry_classification`` / ``get_micro_snapshot`` /
``get_market_snapshot`` 离线读取，用于行业与市值中性化、板块统计等微观维度测试。

默认范围为各票池 ``constituents_*.json`` 的并集——实测与随仓库分发的日K覆盖完全一致
（1977 只），因此微观数据与离线行情严格同域。

用法::

    python scripts/build_offline_micro.py                     # 刷新快照（约 45 次请求）
    python scripts/build_offline_micro.py --universe csi300    # 只刷新单个票池
    python scripts/build_offline_micro.py --as-of 2026-09-11   # 指定快照日期（写入 as_of）

备注：市值/估值为**构建时刻的静态快照**（``as_of`` 列给出日期），离线回测按最新股本
近似；概念板块、上市日期需东财 push2 通道（该接口按 IP 限流，暂不纳入）。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "offline"

# 东财数据中心（datacenter-web 主机，与易被限流的 push2 主机不同）
EM_ORGINFO = "https://datacenter-web.eastmoney.com/api/data/v1/get"
EM_REPORT = "RPT_F10_BASIC_ORGINFO"
EM_COLS = ("SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,EM2016,"
           "INDUSTRYCSRC1,PROVINCE,TRADE_MARKET")
EM_PAGE = 5000        # 全表 24772 条，5000/页 => 5 次请求
EM_MAX_PAGE = 12

# 新浪行业板块（仅兜底）
SINA_HY = "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
SINA_NODE = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             "Market_Center.getHQNodeData?"
             "page={page}&num={num}&sort=symbol&asc=1&node={node}")
SINA_PAGE = 100       # 新浪单页硬上限（num>100 无效）
SINA_MAX_PAGE = 12

TENCENT_Q = "https://qt.gtimg.cn/q={codes}"
BATCH_TX = 50         # 腾讯单次报价只数

SLEEP = 0.35          # 请求间隔
BACKOFF = (2.0, 6.0, 20.0)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
EM_HEADERS = {"User-Agent": UA, "Referer": "https://data.eastmoney.com/"}
SINA_HEADERS = {"User-Agent": UA, "Referer": "http://vip.stock.finance.sina.com.cn/"}
TX_HEADERS = {"User-Agent": UA, "Referer": "https://gu.qq.com/"}

#: 新浪板块中的伪行业（不参与行业归类）
SINA_PSEUDO = {"次新股"}
#: 新浪兜底行业（仅在无具体行业时使用）
SINA_CATCH_ALL = {"其它行业", "综合行业"}

MICRO_FIELDS = ["symbol", "instrument", "name", "board",
                "industry", "industry_l2", "industry_l3", "area",
                "price", "total_mv", "float_mv", "shares_total", "shares_float",
                "pe", "pb", "source", "quote_source", "as_of"]


# ----------------------------------------------------------------------
# 代码工具
# ----------------------------------------------------------------------

def board_of(code: str) -> str:
    """按代码前缀判定市场板块（本地规则，无需联网）。"""
    c = str(code).zfill(6)
    if c.startswith(("688", "689")):
        return "科创板"
    if c.startswith(("300", "301")):
        return "创业板"
    if c.startswith(("920", "43", "82", "83", "87", "88")):
        return "北交所"
    if c.startswith(("600", "601", "603", "605")):
        return "沪市主板"
    if c.startswith(("000", "001", "002", "003")):
        return "深市主板"
    return "其他"


def instrument_of(code: str, board: str) -> str:
    """6 位代码 -> 离线日K使用的 instrument 命名（SH/SZ/BJ 前缀）。"""
    c = str(code).zfill(6)
    if board == "北交所":
        return "BJ" + c
    if board.startswith("沪"):
        return "SH" + c
    if board.startswith("深"):
        return "SZ" + c
    return ("SH" if c[0] in "69" else "SZ") + c


def sina_symbol(code: str) -> str:
    """6 位代码 -> 新浪/腾讯行情代码：600519 -> sh600519。"""
    c = str(code).zfill(6)
    if c.startswith(("920", "43", "82", "83", "87", "88")):
        return "bj" + c
    return ("sh" if c[0] in "69" else "sz") + c


def _get(url: str, headers: Dict[str, str], timeout: int = 30) -> Optional[requests.Response]:
    """带退避重试的 GET，失败返回 None。"""
    err = ""
    for wait in (0.0, *BACKOFF):
        if wait:
            time.sleep(wait)
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
    print(f"    [警告] 请求失败（{err}）：{url[:90]}")
    return None


def _f(val, scale: float = 1.0) -> Optional[float]:
    """安全浮点转换：空值/非数字/非正数一律返回 None（非正市值视为缺失）。"""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or x <= 0:
        return None
    return x * scale


def _split_industry(em2016: str) -> tuple:
    """东财三级行业 ``食品饮料-饮料-白酒`` -> (一级, 二级, 三级)。"""
    parts = [x for x in str(em2016 or "").split("-") if x and x != "-"]
    return (parts[0] if parts else None,
            parts[1] if len(parts) > 1 else None,
            parts[2] if len(parts) > 2 else None)


# ----------------------------------------------------------------------
# 东财公司概况：行业 / 地区 / 名称（主通道）
# ----------------------------------------------------------------------

def fetch_em_orginfo(universe: set) -> Dict[str, dict]:
    """全表分页拉取公司概况，只保留 ``universe`` 内股票。"""
    out: Dict[str, dict] = {}
    pages = EM_MAX_PAGE
    for page in range(1, EM_MAX_PAGE + 1):
        params = {
            "reportName": EM_REPORT, "columns": EM_COLS, "filter": "",
            "pageSize": EM_PAGE, "pageNumber": page,
            "source": "WEB", "client": "WEB",
        }
        r = _get(f"{EM_ORGINFO}?{urlencode(params)}", EM_HEADERS)
        if r is None:
            break
        try:
            res = r.json().get("result") or {}
        except Exception as e:  # noqa: BLE001
            print(f"    [警告] 东财第 {page} 页解析失败：{e}")
            break
        data = res.get("data") or []
        if not data:
            break
        pages = int(res.get("pages") or page)
        for d in data:
            code = str(d.get("SECURITY_CODE") or "").zfill(6)
            if code not in universe or code in out:
                continue
            lv1, lv2, lv3 = _split_industry(d.get("EM2016"))
            out[code] = {
                "name": str(d.get("SECURITY_NAME_ABBR") or "").strip(),
                "industry": lv1,
                "industry_l2": lv2,
                "industry_l3": lv3,
                "area": (str(d.get("PROVINCE")).strip()
                         if d.get("PROVINCE") not in (None, "", "-") else None),
                "source": "eastmoney" if lv1 else None,
            }
        print(f"  [东财 {page}/{pages}] 全表 {len(data)} 条，命中累计 {len(out)}/{len(universe)}")
        if page >= pages:
            break
        time.sleep(SLEEP)
    return out


# ----------------------------------------------------------------------
# 腾讯行情：价 / 市值 / 估值（主通道）
# ----------------------------------------------------------------------

def fetch_tencent_batch(codes: List[str]) -> Dict[str, dict]:
    """腾讯批量报价（<=50 只/次）-> {6位代码: 行情字段}。"""
    url = TENCENT_Q.format(codes=",".join(sina_symbol(c) for c in codes))
    r = _get(url, TX_HEADERS)
    if r is None:
        return {}
    txt = r.content.decode("gbk", "ignore")
    out: Dict[str, dict] = {}
    for line in txt.split(";"):
        if "=" not in line:
            continue
        f = line.split("=", 1)[1].strip().strip('"').split("~")
        if len(f) < 47:
            continue
        code = str(f[2]).zfill(6)
        if code not in codes:
            continue
        out[code] = {
            "name": f[1].strip() or None,
            "price": _f(f[3]),
            "pe": _f(f[39]),                  # 市盈率(TTM)
            "float_mv": _f(f[44], 1e8),       # 单位：亿元
            "total_mv": _f(f[45], 1e8),       # 单位：亿元
            "pb": _f(f[46]),
            "quote_source": "tencent",
        }
    return out


def fill_quotes(codes: List[str], table: Dict[str, dict]) -> int:
    """批量补齐价/市值/估值，返回补齐只数。"""
    if not codes:
        return 0
    print(f"  腾讯行情：{len(codes)} 只（{math.ceil(len(codes) / BATCH_TX)} 批）")
    filled = 0
    for i in range(0, len(codes), BATCH_TX):
        batch = codes[i:i + BATCH_TX]
        got = fetch_tencent_batch(batch)
        for code, rec in got.items():
            row = table.setdefault(code, {})
            for k, v in rec.items():
                if k == "name":
                    row.setdefault("name", v)
                else:
                    row[k] = v
            filled += 1
        print(f"    [{i // BATCH_TX + 1}/{math.ceil(len(codes) / BATCH_TX)}] "
              f"请求 {len(batch)} 只，返回 {len(got)} 只（累计 {filled}/{len(codes)}）")
        time.sleep(SLEEP)
    return filled


# ----------------------------------------------------------------------
# 新浪行业板块（兜底通道）
# ----------------------------------------------------------------------

def fetch_sina_industries() -> List[tuple]:
    """新浪行业板块清单 -> [(板块代码, 板块名)]。"""
    r = _get(SINA_HY, SINA_HEADERS)
    if r is None:
        return []
    try:
        raw = json.loads(r.content.decode("gbk", "ignore").split("=", 1)[1].strip().rstrip(";"))
    except Exception as e:  # noqa: BLE001
        print(f"    [警告] 新浪行业列表解析失败：{e}")
        return []
    return [(c, str(v).split(",")[1]) for c, v in raw.items() if len(str(v).split(",")) > 2]


def fetch_sina_members(hy_code: str) -> List[dict]:
    """取单个新浪行业板块的成分股（逐页翻到底，单页上限 100）。"""
    rows: List[dict] = []
    for page in range(1, SINA_MAX_PAGE + 1):
        r = _get(SINA_NODE.format(page=page, num=SINA_PAGE, node=hy_code), SINA_HEADERS)
        if r is None:
            break
        try:
            data = json.loads(r.content.decode("gbk", "ignore"))
        except Exception:  # noqa: BLE001
            break
        if not isinstance(data, list) or not data:
            break
        rows.extend(data)
        if len(data) < SINA_PAGE:
            break
        time.sleep(SLEEP)
    return rows


def fill_industry_from_sina(todo: List[str], table: Dict[str, dict]) -> int:
    """用新浪行业板块补齐缺行业的股票（顺带补价与市值）。"""
    want = set(todo)
    inds = fetch_sina_industries()
    if not inds:
        print("  [警告] 新浪行业清单为空，行业兜底跳过")
        return 0
    print(f"  新浪兜底：{len(inds)} 个板块，待补 {len(todo)} 只股票")
    specific: Dict[str, dict] = {}
    catch_all: Dict[str, dict] = {}
    for i, (hy_code, hy_name) in enumerate(inds, 1):
        if hy_name in SINA_PSEUDO:
            continue
        rows = fetch_sina_members(hy_code)
        target = catch_all if hy_name in SINA_CATCH_ALL else specific
        hit = 0
        for row in rows:
            code = str(row.get("code", "")).zfill(6)
            if code not in want or code in specific or code in catch_all:
                continue
            hit += 1
            target[code] = {
                "industry": hy_name,
                "price": _f(row.get("trade")),
                "total_mv": _f(row.get("mktcap"), 1e4),   # 新浪单位：万元
                "float_mv": _f(row.get("nmc"), 1e4),      # 新浪单位：万元
                "pe": _f(row.get("per")),
                "pb": _f(row.get("pb")),
                "source": "sina",
                "quote_source": "sina",
            }
        if hit or i % 10 == 0:
            print(f"    [{i}/{len(inds)}] {hy_name}: 命中 {hit} 只"
                  f"（累计 {len(specific) + len(catch_all)}/{len(todo)}）")
        time.sleep(SLEEP)
    for code, rec in catch_all.items():     # 兜底行业只在无具体行业时生效
        specific.setdefault(code, rec)
    for code, rec in specific.items():
        row = table.setdefault(code, {})
        for k, v in rec.items():            # 只补空位，不覆盖已有值
            if row.get(k) is None:
                row[k] = v
    return len(specific)


# ----------------------------------------------------------------------
# 组装 / 落盘
# ----------------------------------------------------------------------

def to_micro(table: Dict[str, dict], universe: List[str], as_of: str) -> pd.DataFrame:
    """把抓取结果整理成快照表（含本地判定的板块与股本推导）。"""
    rows = []
    for code in universe:
        rec = table.get(code, {})
        board = board_of(code)
        price = rec.get("price")
        total_mv = rec.get("total_mv")
        float_mv = rec.get("float_mv")
        rows.append({
            "symbol": code,
            "instrument": instrument_of(code, board),
            "name": rec.get("name") or "",
            "board": board,
            "industry": rec.get("industry"),
            "industry_l2": rec.get("industry_l2"),
            "industry_l3": rec.get("industry_l3"),
            "area": rec.get("area"),
            "price": price,
            "total_mv": total_mv,
            "float_mv": float_mv,
            "shares_total": (total_mv / price) if (total_mv and price) else None,
            "shares_float": (float_mv / price) if (float_mv and price) else None,
            "pe": rec.get("pe"),
            "pb": rec.get("pb"),
            "source": rec.get("source") or None,
            "quote_source": rec.get("quote_source") or None,
            "as_of": as_of,
        })
    df = pd.DataFrame(rows, columns=MICRO_FIELDS)
    df = df.sort_values("symbol").reset_index(drop=True)
    for col in ("price", "total_mv", "float_mv", "shares_total",
                "shares_float", "pe", "pb"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def load_universe(pool: Optional[str] = None) -> List[str]:
    """票池成分股并集（6 位裸代码）；``pool`` 指定时只取该票池。"""
    paths = ([OUT_DIR / f"constituents_{pool}.json"] if pool
             else sorted(OUT_DIR.glob("constituents_*.json")))
    codes: set = set()
    for p in paths:
        if not p.exists():
            print(f"  [警告] 票池文件不存在：{p}")
            continue
        for inst in json.loads(p.read_text(encoding="utf-8")):
            inst = str(inst).strip().upper()
            codes.add(inst[2:] if inst[:2] in ("SH", "SZ", "BJ") else inst.zfill(6))
    return sorted(codes)


def write_micro(df: pd.DataFrame, as_of: str) -> Dict[str, object]:
    """写出 parquet 并返回 meta.json 的 micro 段。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "micro_snapshot.parquet", index=False)
    return {
        "micro": {
            "file": "micro_snapshot.parquet",
            "as_of": as_of,
            "symbols": len(df),
            "industries": {
                "level1": int(df["industry"].nunique(dropna=True)),
                "level2": int(df["industry_l2"].nunique(dropna=True)),
                "areas": int(df["area"].nunique(dropna=True)),
            },
            "coverage": {
                "industry": int(df["industry"].notna().sum()),
                "industry_l2": int(df["industry_l2"].notna().sum()),
                "area": int(df["area"].notna().sum()),
                "total_mv": int(df["total_mv"].notna().sum()),
                "pe": int(df["pe"].notna().sum()),
                "pb": int(df["pb"].notna().sum()),
            },
            "boards": {str(k): int(v) for k, v in df["board"].value_counts().items()},
            "sources": {str(k): int(v) for k, v in df["source"].value_counts(dropna=False).items()},
            "quote_sources": {str(k): int(v)
                              for k, v in df["quote_source"].value_counts(dropna=False).items()},
            "generator": "scripts/build_offline_micro.py",
            "note": ("行业/地区来自东财公司概况报表（东财三级行业、注册地省份），"
                     "价格与市值/估值为构建时刻的腾讯行情静态快照（市值单位元）；"
                     "市场板块由代码前缀本地判定。概念板块、上市日期需东财 push2 通道，"
                     "该接口按 IP 限流，暂未纳入。"),
        }
    }


def update_meta(patch: Dict[str, object]) -> None:
    meta_path = OUT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(patch)
    meta["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已更新 meta.json：{sorted(patch)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 data/offline/micro_snapshot.parquet")
    ap.add_argument("--universe", default=None,
                    help="票池名（csi300/csi800/...）；默认取 constituents_*.json 并集")
    ap.add_argument("--as-of", default=pd.Timestamp.now().strftime("%Y-%m-%d"),
                    help="快照日期（写入 as_of 列与 meta）")
    ap.add_argument("--skip-quote", action="store_true", help="只取行业/板块，不取行情市值")
    args = ap.parse_args()

    universe = load_universe(args.universe)
    if not universe:
        raise SystemExit(f"{OUT_DIR} 下未找到任何 constituents_*.json，无法确定票池")
    print(f"票池 {args.universe or '全部并集'}：{len(universe)} 只")

    table = fetch_em_orginfo(set(universe))
    n_ind = sum(1 for r in table.values() if r.get("industry"))
    print(f"东财通道：行业 {n_ind}/{len(universe)} 只")

    missing_ind = [c for c in universe if not table.get(c, {}).get("industry")]
    if missing_ind:
        fill_industry_from_sina(missing_ind, table)

    if not args.skip_quote:
        need = [c for c in universe if table.get(c, {}).get("total_mv") is None]
        fill_quotes(need, table)

    df = to_micro(table, universe, args.as_of)
    patch = write_micro(df, args.as_of)

    print(f"快照 {len(df)} 行：行业 {patch['micro']['coverage']['industry']} 只 / "
          f"市值 {patch['micro']['coverage']['total_mv']} 只 / "
          f"一级行业 {patch['micro']['industries']['level1']} 个 / "
          f"二级行业 {patch['micro']['industries']['level2']} 个")
    print("板块分布:", patch["micro"]["boards"])
    print("行业来源:", patch["micro"]["sources"], "行情来源:", patch["micro"]["quote_sources"])
    for col in ("industry", "total_mv"):
        miss = df.loc[df[col].isna(), "symbol"].tolist()
        if miss:
            print(f"  [提示] {col} 缺失 {len(miss)} 只（前 10：{miss[:10]}）")

    update_meta(patch)
    print(f"已写出 {OUT_DIR / 'micro_snapshot.parquet'}")


if __name__ == "__main__":
    main()
