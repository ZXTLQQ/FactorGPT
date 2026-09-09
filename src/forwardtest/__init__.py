"""forwardtest — Headline Arena 前瞻检验桥接层。

因子验证（IC 分析等）本质是历史检验，回答不了"这个因子观点向前看还成立吗"。
本包把 FactorGPT 因子层的宏观观点（利率、风格、商品方向）转成每日概率预测，
提交到 Headline Arena（https://headlinearena.com），得到一条与回测相互独立的
forward（前瞻）检验线：预测在结果出现之前锁定，结算标准在出题时冻结，由第三方
按真实行情机械结算（方向预测按 50 + confidence*50 计分，宏观数值按 CRPS）。

设计原则与 FactorGPT 全项目一致：
- 零第三方依赖（纯 Python stdlib HTTP），CI / 离线环境均可导入；
- 默认 dry_run（只写本地账本），真实提交必须显式 --live 且具备凭据；
- 任何网络 / 凭据故障都优雅降级为本地影子记录，绝不中断因子流水线。

包结构：
- client.py     HeadlineArenaClient：注册 / 鉴权 / scope / 挑战发现 / 提交 / 结果 / 校准
- translator.py 宏观观点 -> HA 预测的翻译器（关键词主题匹配 + 资产映射）
- ledger.py     ForwardLedger：预测在结果存在前锁定，结算后由外部回填（不可篡改预测字段）
- scorecard.py  前瞻评分卡：准确率 / Brier / 校准曲线 / HA 官方分，与回测完全独立
- runner.py     编排：run(每日循环) / settle(结算回填) / scorecard(生成报告)
"""

from __future__ import annotations

__version__ = "0.1.0"
