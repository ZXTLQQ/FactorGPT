"""
非结构化数据因子挖掘 (src/engine/unstructured_miner.py)

从非结构化数据源中提取量化因子信号：
1. 文本数据 — 新闻/研报/公告的情感/实体/话题因子
2. PDF/Excel/CSV/JSON 用户上传 — 自动解析并转为因子时序
3. 另类数据 — 社交媒体热词/搜索热度/供应链关系/
4. 因子融合 — 将非结构化信号与价格-成交量因子结合

零外部依赖核心解析（pandas + re + json），扩展依赖（PyPDF2/openpyxl）可选安装。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ===================================================================
# 1. 文本分析引擎（零外部 NLP 依赖，基于关键词 + 规则）
# ===================================================================

# 中文情感词典（轻量级，覆盖常见的量化相关词汇）
_SENTIMENT_LEXICON = {
    "positive": [
        "增长", "盈利", "突破", "创新高", "利好", "超预期", "分红",
        "回购", "增持", "扩张", "中标", "订单", "投产", "注册",
        "通过", "获批", "解除", "恢复", "上升", "改善", "升级",
        "龙头", "领先", "冠军", "翻倍", "涨停", "加仓", "买入",
        "增资", "盈利改善", "业绩预增", "扭亏", "大幅增长",
    ],
    "negative": [
        "下跌", "亏损", "爆雷", "违约", "诉讼", "处罚", "调查",
        "ST", "*ST", "退市", "减持", "冻结", "质押", "逾期",
        "停产", "召回", "事故", "损失", "下降", "恶化", "暴跌",
        "跌停", "减仓", "卖出", "利空", "低于预期", "业绩预减",
        "商誉减值", "资产减值", "担保", "违规", "停牌",
    ],
    "uncertainty": [
        "可能", "预计", "或将", "有待", "风险", "不确定", "波动",
        "变化", "调整", "变更", "重组", "收购", "转让",
        "注资", "引入战投", "拟", "筹划", "尚需", "如需",
    ],
}

# 行业关键词映射
_INDUSTRY_KEYWORDS = {
    "新能源": ["光伏", "风电", "储能", "锂电池", "宁德", "比亚迪", "隆基"],
    "半导体": ["芯片", "晶圆", "光刻", "中芯", "海思", "紫光", "韦尔"],
    "医药": ["创新药", "药明", "恒瑞", "疫苗", "原料药", "医疗器械"],
    "消费": ["茅台", "五粮液", "伊利", "海天", "美的", "格力", "海尔"],
    "金融": ["银行", "证券", "保险", "平安", "招商", "中信"],
    "AI/TMT": ["人工智能", "大模型", "ChatGPT", "算法", "算力", "讯飞", "商汤"],
}

_COMMON_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
}


class TextAnalyzer:
    """纯规则的中文文本分析器（无需NLP模型）。"""

    def __init__(self) -> None:
        self._lexicon = _SENTIMENT_LEXICON
        self._industry_map = _INDUSTRY_KEYWORDS
        self._stopwords = _COMMON_STOPWORDS

    def analyze(self, text: str) -> Dict[str, Any]:
        """分析单条文本。

        Returns:
            {"sentiment": float,       # -1 到 1
             "pos_hits": int,
             "neg_hits": int,
             "uncertainty_hits": int,
             "industries": [str, ...],
             "entities": [str, ...],
             "word_count": int}
        """
        text_lower = text.lower()
        pos_hits = sum(1 for w in self._lexicon["positive"] if w in text)
        neg_hits = sum(1 for w in self._lexicon["negative"] if w in text)
        unc_hits = sum(1 for w in self._lexicon["uncertainty"] if w in text)

        # 情感分数: 正/负比例差，经 sigmoid 平滑
        raw = pos_hits - neg_hits
        sentiment = np.tanh(raw / max(pos_hits + neg_hits + 1, 1))

        # 行业识别
        industries = []
        for ind, kws in self._industry_map.items():
            if any(kw in text for kw in kws):
                industries.append(ind)

        # 实体提取（简单：大写/中文专有名词片段）
        entities = re.findall(r'[\u4e00-\u9fff]{2,6}(?:公司|集团|股份|科技|医药|银行|证券|基金)', text)
        # 也提取英文大写缩写
        entities += re.findall(r'[A-Z]{2,6}', text)

        words = [w for w in re.findall(r'[\u4e00-\u9fff]+', text) if w not in self._stopwords]

        return {
            "sentiment": round(sentiment, 4),
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "uncertainty_hits": unc_hits,
            "industries": list(set(industries)),
            "entities": list(set(entities)),
            "word_count": len(words),
            "top_words": words[:20],
        }

    def analyze_batch(
        self, texts: pd.Series, group_col: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """批量分析，返回 DataFrame。group_col 用于按 symbol 分组。"""
        results = [self.analyze(t) for t in texts]
        df = pd.DataFrame(results)
        if group_col is not None:
            df["symbol"] = group_col.values
        return df


# ===================================================================
# 2. 用户数据上传解析器
# ===================================================================

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".txt", ".pdf"}


class DataUploadParser:
    """解析用户上传的非结构化/结构化数据文件。

    支持格式：CSV(.csv), Excel(.xlsx/.xls), JSON(.json), 文本(.txt), PDF(.pdf)
    自动检测列映射：date, symbol, factor/price/volume 等
    """

    COLUMN_ALIASES: Dict[str, List[str]] = {
        "date": ["date", "trade_date", "trading_day", "date", "time", "datetime", "timestamp", "日期", "交易日"],
        "symbol": ["symbol", "code", "ticker", "stock_code", "股票", "代码", "标的"],
        "close": ["close", "price", "收盘价", "收盘", "最新价"],
        "open": ["open", "开盘价", "开盘"],
        "high": ["high", "最高价", "最高"],
        "low": ["low", "最低价", "最低"],
        "volume": ["volume", "vol", "成交量", "量"],
        "amount": ["amount", "turnover", "成交额", "金额"],
        "pct_chg": ["pct_chg", "pct_change", "ret", "chg", "涨跌幅", "收益率"],
        "text": ["text", "content", "新闻", "公告", "研报", "body", "article", "description"],
        "factor": ["factor", "alpha", "signal", "score", "value", "因子", "信号", "评分"],
    }

    @staticmethod
    def _detect_column_mapping(columns: List[str]) -> Dict[str, str]:
        """自动检测列名到标准字段的映射。"""
        mapping: Dict[str, str] = {}
        lower_cols = {c: c.lower() for c in columns}
        for std_name, aliases in DataUploadParser.COLUMN_ALIASES.items():
            for alias in aliases:
                alias_lower = alias.lower()
                for col_orig, col_lower in lower_cols.items():
                    if alias_lower == col_lower:
                        mapping.setdefault(std_name, col_orig)
                        break
                if std_name in mapping:
                    break
        return mapping

    def parse_file(
        self, file_path: Union[str, Path], sheet_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """解析文件并返回 (数据框, 元信息字典)。

        元信息包含：file_type, columns, column_mapping, shape, n_symbols 等。
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式：{ext}，支持：{SUPPORTED_EXTENSIONS}")

        # 读入
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            df = pd.DataFrame({"text": lines, "line_no": range(len(lines))})
        elif ext == ".pdf":
            df = self._parse_pdf(file_path)
        else:
            df = pd.DataFrame()

        # 列映射
        mapping = self._detect_column_mapping(list(df.columns))

        # 元信息
        meta = {
            "file_name": file_path.name,
            "file_type": ext,
            "columns": list(df.columns),
            "column_mapping": mapping,
            "shape": df.shape,
            "n_symbols": df[mapping["symbol"]].nunique() if "symbol" in mapping else 0,
            "date_range": (
                (str(df[mapping["date"]].min()), str(df[mapping["date"]].max()))
                if "date" in mapping else None
            ),
        }

        return df, meta

    def _parse_pdf(self, file_path: Path) -> pd.DataFrame:
        """PDF 解析（尝试多种策略）。"""
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                texts = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        texts.append(t)
            return pd.DataFrame({"text": texts, "page": range(1, len(texts) + 1)})
        except ImportError:
            pass

        # 回退：按行读取二进制
        with open(file_path, "rb") as f:
            raw = f.read()
        # 尝试解码
        for enc in ["utf-8", "gbk", "latin-1"]:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = str(raw)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return pd.DataFrame({"text": lines, "line_no": range(len(lines))})

    def to_factor_time_series(
        self, df: pd.DataFrame, mapping: Dict[str, str], factor_expr: Optional[str] = None
    ) -> pd.DataFrame:
        """将上传数据转为标准因子时序（date, symbol, factor）。"""
        if "date" not in mapping or "symbol" not in mapping:
            raise ValueError("必须包含 date 列和 symbol 列（或对应别名）")

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[mapping["date"]])
        out["symbol"] = df[mapping["symbol"]].astype(str)

        if factor_expr:
            out["factor"] = eval(factor_expr, {"df": df, "np": np})
        elif "factor" in mapping:
            out["factor"] = pd.to_numeric(df[mapping["factor"]], errors="coerce")
        elif "close" in mapping:
            out["factor"] = pd.to_numeric(df[mapping["close"]], errors="coerce")
        elif "text" in mapping:
            analyzer = TextAnalyzer()
            sentiments = analyzer.analyze_batch(df[mapping["text"]])
            out["factor"] = sentiments["sentiment"].values
        else:
            raise ValueError("无法推断因子值列，请指定 factor_expr 参数")

        out["factor"] = out["factor"].astype(float).fillna(0.0)
        return out.sort_values(["symbol", "date"])


# ===================================================================
# 3. 另类数据源管理器
# ===================================================================

class AlternativeDataManager:
    """另类数据源接入与管理。

    支持的数据源类型：
    - social_media: 社交媒体热词/讨论热度
    - search_trends: 搜索热度指数
    - supply_chain: 供应链关系数据
    - satellite: 卫星图像特征（夜间灯光、工厂活跃度等）
    - credit_card: 消费刷卡数据聚合
    - shipping: 航运/物流追踪数据
    - weather: 气象灾害对区域经济影响

    当前回退到 CSV/JSON 本地文件模式，后续可扩展为 API 接入。
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "alternative"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._parser = DataUploadParser()

    def register_source(
        self,
        source_id: str,
        source_type: str,
        file_path: Optional[str] = None,
        api_config: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """注册一个另类数据源。"""
        self._sources[source_id] = {
            "type": source_type,
            "file_path": file_path,
            "api_config": api_config or {},
            "meta": meta or {},
            "registered_at": datetime.now().isoformat(),
        }

    def list_sources(self) -> Dict[str, Dict[str, Any]]:
        """列出所有已注册的数据源。"""
        return dict(self._sources)

    def load_source(
        self, source_id: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """加载一个数据源的数据。"""
        if source_id not in self._sources:
            raise KeyError(f"未注册的数据源：{source_id}")

        src = self._sources[source_id]
        if src["file_path"]:
            return self._parser.parse_file(src["file_path"])
        raise ValueError(f"数据源 {source_id} 无本地文件路径且暂不支持 API 调用")

    def source_to_factor(
        self,
        source_id: str,
        factor_expr: Optional[str] = None,
    ) -> pd.DataFrame:
        """将另类数据源转为因子时序。"""
        df, meta = self.load_source(source_id)
        return self._parser.to_factor_time_series(
            df, meta["column_mapping"], factor_expr=factor_expr
        )

    def derive_sentiment_factor(
        self, source_id: str, text_column: str = "text"
    ) -> pd.DataFrame:
        """从文本类数据源中派生情感因子。"""
        df, meta = self.load_source(source_id)
        if text_column not in df.columns:
            raise ValueError(f"列 {text_column} 不在文件中: {meta['columns']}")

        analyzer = TextAnalyzer()
        results = analyzer.analyze_batch(df[text_column])

        out = pd.DataFrame()
        if "date" in meta["column_mapping"]:
            out["date"] = pd.to_datetime(df[meta["column_mapping"]["date"]])
        if "symbol" in meta["column_mapping"]:
            out["symbol"] = df[meta["column_mapping"]["symbol"]].astype(str)
        out["factor"] = results["sentiment"].values
        out["pos_hits"] = results["pos_hits"].values
        out["neg_hits"] = results["neg_hits"].values

        return out

    def derive_industry_exposure_factor(
        self, source_id: str, industry: str, text_column: str = "text"
    ) -> pd.DataFrame:
        """从文本数据中派生特定行业的曝光度因子。"""
        df, meta = self.load_source(source_id)
        if text_column not in df.columns:
            raise ValueError(f"列 {text_column} 不在文件中")

        analyzer = TextAnalyzer()
        kws = analyzer._industry_map.get(industry, [])
        count = df[text_column].apply(lambda t: sum(1 for kw in kws if kw in str(t)))

        out = pd.DataFrame()
        if "date" in meta["column_mapping"]:
            out["date"] = pd.to_datetime(df[meta["column_mapping"]["date"]])
        if "symbol" in meta["column_mapping"]:
            out["symbol"] = df[meta["column_mapping"]["symbol"]].astype(str)
        out["factor"] = count.values / max(count.max(), 1)

        return out


# ===================================================================
# 4. 非结构化因子融合（与价格/量价因子整合）
# ===================================================================

class UnstructuredFactorIntegrator:
    """将非结构化数据因子与结构化量价因子融合。

    融合策略：
    - naive_mean: 简单均值融合
    - ranked_ensemble: 分别排名后取均值
    - weighted: 按预设权重加权（可来自回测IC）
    - orthogonalized: 对现有因子正交化后叠加（减少共线性）
    """

    def __init__(self) -> None:
        self._factors: Dict[str, pd.DataFrame] = {}
        self._weights: Dict[str, float] = {}

    def add_factor(self, name: str, df: pd.DataFrame, weight: float = 1.0) -> None:
        """添加一个因子（需含 date, symbol, factor 列）。"""
        self._factors[name] = df.copy()
        self._weights[name] = weight

    def fuse(
        self,
        method: str = "ranked_ensemble",
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """融合所有已添加的因子。

        Args:
            method: 融合方式 — 'naive_mean', 'ranked_ensemble', 'weighted', 'orthogonalized'
            weights: 手动指定权重（覆盖默认权重）

        Returns:
            含 date, symbol, factor 列的 DataFrame
        """
        if not self._factors:
            raise ValueError("请先添加因子")

        # 对齐所有因子到统一的 date × symbol 面板
        panels = []
        factor_names = list(self._factors.keys())
        for name in factor_names:
            p = self._factors[name][["date", "symbol", "factor"]].copy()
            p = p.rename(columns={"factor": name})
            panels.append(p)

        merged = panels[0]
        for p in panels[1:]:
            merged = merged.merge(p, on=["date", "symbol"], how="outer")
        merged = merged.fillna(0.0)

        factor_cols = [n for n in factor_names]
        actual_weights = weights or self._weights

        if method == "naive_mean":
            merged["factor"] = merged[factor_cols].mean(axis=1)
        elif method == "ranked_ensemble":
            ranked = merged[factor_cols].rank(pct=True, axis=0)
            merged["factor"] = ranked.mean(axis=1)
        elif method == "weighted":
            w_sum = sum(actual_weights.get(n, 0) for n in factor_cols) or 1.0
            merged["factor"] = sum(
                merged[n] * actual_weights.get(n, 0) for n in factor_cols
            ) / w_sum
        elif method == "orthogonalized":
            # 简单正交化：用第一个因子回归其他因子，取残差再融合
            base = merged[factor_cols[0]].values
            ortho_factors = [base]
            for n in factor_cols[1:]:
                resid = merged[n].values - np.polyval(np.polyfit(base, merged[n].values, 1), base)
                ortho_factors.append(resid)
            stacked = np.column_stack(ortho_factors)
            merged["factor"] = np.nanmean(stacked, axis=1)

        result = merged[["date", "symbol", "factor"]].copy()
        result["factor"] = result["factor"].fillna(0.0).astype(float)
        return result
