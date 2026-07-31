"""Kronos 金融时序预测基础模型集成包 (src/kronos)。

Kronos (morrisluo/kronos, HuggingFace: NeoQuasar/Kronos-*) 是 decoder-only 的 K 线
序列预测基础模型, 用于预测未来 OHLCV。它**不是**聊天 LLM, 因此本项目把它作为
"预测因子" 集成: 用其未来收益预测构造选股信号, 代码生成仍由 config.llm 中的
OpenAI 兼容模型(可指向本地 Ollama)负责。

由于 Kronos 依赖 torch + transformers + KronosPredictor 且权重需从 HuggingFace
下载, 本模块在以下情况自动降级为 stub(纯几何动量代理) 并告警, 保证流水线离线 /
轻量环境也能跑通:
  - 未安装 KronosPredictor / torch / transformers
  - 权重下载失败
  - config.kronos.fallback_to_stub == true
"""

from .forecaster import KronosForecaster, attach_kronos_factor, KRONOS_COLS

__all__ = ["KronosForecaster", "attach_kronos_factor", "KRONOS_COLS"]
