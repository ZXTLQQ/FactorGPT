---
title: FactorGPT
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: static
pinned: true
license: mit
short_description: AI-Driven Quantitative Factor Discovery and Mining Platform
---

# FactorGPT

AI-Driven Quantitative Factor Discovery and Mining Platform for A-Share Markets.

Describe your investment idea in natural language and watch the LLM Agent turn it
into validated, backtested alpha factors — powered by LangGraph orchestration, a
six-stage factor refinery pipeline, and a 62-factor built-in knowledge base.

## Try It Online

- **Live demo (static build)**: https://zxtlqq-factorgpt-demo.static.hf.space
- **GitHub Repository**: https://github.com/ZXTLQQ/FactorGPT

## What's Inside

- **LLM factor mining agent** — describe an investment idea in natural language; a
  LangGraph loop retrieves factor literature, generates code, sandbox-validates it
  (whitelist imports + AST lookahead detection), backtests, and reflects on the metrics.
- **Six-stage factor refinery** — ore warehouse → mining layer → grinding →
  three-tier screening → alloy blending → methodology report.
- **62-factor built-in library** — name, category, formula (Markdown + LaTeX) and a
  reference Pandas implementation for every factor.
- **Typed factor research layer** (`src/mining/`) — 60 operators in a typed expression DSL,
  a point-in-time safe panel, a four-dimension evaluator (data quality / predictive power /
  stability / correlation) with turnover and expression-complexity penalties, factor risk &
  crowding gates, and a Markdown report that pins every number to its evidence.
- **AI factor-system advisor** — one backtest is compressed into a single fact table that a
  local rule engine and an LLM both answer from; the model may not cite a number that is not
  in the table, and degrades to the rule answer when it fails instead of going blank.
- **Spectral cleaning of the factor correlation matrix** — random-matrix noise bounds,
  cleaning, weight solving and per-factor risk decomposition, rendered in the factor-system
  page and read straight into the advisor's diagnosis.
- **Bundled offline dataset** — qfq daily bars for the CSI 800 pool ship with the
  repository, so backtests run with no network and no API keys.
- **Independent forward testing** — factor-layer macro views become
  locked-before-outcome predictions that a third party settles and scores; a check that
  historical IC backtests structurally cannot provide.

## Data Sources

- Bundled offline dataset (default; no network, no key)
- AKShare / Tushare / Baostock self-crawled feeds (fallback chain)
- NeoData platform service (natural-language market Q&A)
- EastMoney Miaoxiang (妙想) MX API: market data, news search, smart stock screening,
  watchlist management, simulated portfolio, and financial community content
  (see `factorgpt-skill/skills/` in the repository; configure your own `MX_APIKEY`)

## Documentation

- [Repository README](https://github.com/ZXTLQQ/FactorGPT#readme) — full feature list,
  architecture, configuration and deployment
- [Ablation study](https://github.com/ZXTLQQ/FactorGPT/blob/main/docs/ablation_report.md)
  — per-module out-of-sample contribution (ΔICIR)
- [Typed factor research layer](https://github.com/ZXTLQQ/FactorGPT#10-typed-factor-research-layer-srcmining)
  — expression DSL, PIT-safe panel and operator grid miner
