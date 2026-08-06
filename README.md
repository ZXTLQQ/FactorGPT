# FactorGPT — LLM-Powered Quantitative Factor Mining & Industrialization Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)](https://hub.docker.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![HF Spaces](https://img.shields.io/badge/🤗%20Demo-Online-ff9a00?logo=huggingface)](https://huggingface.co/spaces/ZXTLQQ/factorgpt-demo)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/ZXTLQQ/FactorGPT)

**FactorGPT** is an LLM-powered intelligent financial factor industrialization platform that deeply integrates natural language understanding with quantitative finance factor engineering. It supports automated factor extraction, validation, combination optimization, and production-grade deployment from both structured and unstructured data sources — all driven by natural language commands.

> **Keywords**: Quantitative Finance, Alpha Factor Mining, LLM Agent, Factor Backtesting, Factor Library, Genetic Programming, Reinforcement Learning, Alternative Data, Streamlit, A-Share, Financial AI, FactorGPT, Factor Refinery, RPN Engine, IC Analysis, Multi-factor Model, LangGraph, Python Quant

---

## What Problem Does FactorGPT Solve?

Traditional quantitative factor research faces three major pain points: **high barrier to entry** (requires extensive domain expertise and programming skills), **slow iteration cycles** (manual factor design, coding, and backtesting loops take days to weeks), and **siloed workflows** (data, generation, evaluation, and deployment are disconnected).

FactorGPT addresses these challenges by:

1. **Natural Language to Factor Code**: Describe your investment idea in plain language, and the LLM Agent generates, validates, and backtests factor code automatically. The system includes a built-in knowledge base of 61 traditional factors as reference context.

2. **End-to-End Industrial Pipeline**: The "Six-Stage Factor Refinery" mimics an industrial smelting process — from raw data ore to finished factor products — with built-in quality control at every stage.

3. **Safety and Reliability**: A sandboxed execution environment with white-listed imports, lookahead bias detection (AST-based), automatic winsorization/neutralization/standardization, and out-of-sample validation ensures production-ready factor quality.

4. **Multi-Model Support**: Works with DeepSeek, OpenAI, Qwen, local Ollama models, and any OpenAI-compatible endpoint. The system auto-degrades gracefully when dependencies are missing, ensuring it runs in any environment.

---

## 🚀 Try It Online — No Installation

<p align="center">
  <a href="https://huggingface.co/spaces/ZXTLQQ/factorgpt-demo">
    <img src="https://img.shields.io/badge/🤗-Open_in_HuggingFace_Spaces-ff9a00?style=for-the-badge&logo=huggingface" alt="HuggingFace Spaces">
  </a>
</p>

FactorGPT provides a free online demo on HuggingFace Spaces — **no API keys, no sign-up, no installation required**. Describe your investment idea in natural language and see the factor go from code generation to backtest results in seconds. The demo includes: factor mining, 61-factor library browser, interactive IC charts, and a six-stage refinery pipeline walkthrough.

---

## Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **OS**: Linux, macOS, or Windows
- **Memory**: 8GB RAM minimum (16GB recommended for Transformer/RL modules)
- **Disk**: ~5GB for dependencies + model weights

### Option 1: Docker One-Click Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/factor-gpt/factor-gpt.git
cd factor-gpt

# Build and start
docker-compose up -d

# Open browser at http://localhost:8501
```

The Docker image comes pre-configured with all dependencies, synthetic sample data, and a fully offline-capable demonstration environment. No API keys or network access required.

### Option 2: Local Installation

```bash
# 1. Clone and enter directory
git clone https://github.com/factor-gpt/factor-gpt.git
cd factor-gpt

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies (locked versions for reproducibility)
pip install -r requirements.lock.txt

# 4. (Optional) Prefetch real market data for offline use
python scripts/prefetch_data.py

# 5. Launch the web interface
streamlit run src/ui/app.py
```

Open your browser at `http://localhost:8501` to access the 17-page integrated web dashboard.

### Quick Test (No Network Required)

```bash
# Single factor mining (offline, auto-fallback to synthetic data)
python run_agent.py "Build a 20-day momentum factor"

# Six-stage refinery pipeline (offline demo)
python run_agent.py --refinery "Mix daily and monthly frequency, combine short-term reversal with liquidity"

# Run preflight check (10-second health check for offline presentation readiness)
python scripts/preflight_check.py
```

---

## Project Highlights

### 1. Intelligent Factor Mining Agent (LangGraph Orchestration)

The core Agent follows a **Retrieve → Generate → Validate → Evaluate → Reflect** closed loop:

- **Knowledge Retrieval**: Searches factor knowledge base (ChromaDB + BGE embeddings, or jieba fallback) for relevant academic factor literature as LLM context
- **Factor Generation**: LLM generates factor code following strict safety protocols (`alpha_factor(df) -> DataFrame[date, symbol, factor]`)
- **Sandbox Validation**: Secure execution in isolated subprocess with timeout/memory limits, whitelist imports, and AST-based lookahead bias detection
- **Factor Post-processing**: Winsorization → Industry/Market-cap Neutralization → Standardization
- **Backtest Evaluation**: IC, RankIC, ICIR, IC positivity ratio, quantile returns, long-short Sharpe/MDD, turnover, coverage
- **Reflection & Improvement**: If IC threshold not met, LLM reflects on backtest metrics and iteratively improves the factor definition

### 2. Six-Stage Factor Refinery (Industrial Pipeline)

An end-to-end factor production line modeled after industrial smelting:

| Stage | Process | Component | Purpose |
|-------|---------|-----------|---------|
| PART-01 | Ore Warehouse | `FeatureForge` | 28 raw features + 50+ time-series/cross-sectional factor pools, multi-process parallel construction |
| PART-02 | Mining Layer | `TransformerEncoder` + `FactorRLSearch` + LLM | Transformer (d_model=128, 2 layers, 5 heads) vectorization + MaskablePPO factor combination search + LLM vein exploration |
| PART-03 | Grinding Workshop | `RPNEngine` | Rank IC/IR/ICIR quantification + stability assessment + parallel batch evaluation |
| PART-04 | Three-Tier Screening | `Screener` | LASSO de-redundancy → Human-AI collaborative review → TOP 10% cutoff |
| PART-05 | Alloy Blending | `AlphaPool` | ICIR-weighted + orthogonalization synthesis + leave-one-out overfitting test |
| PART-06 | Methodology Report | `MethodologyReport` | Automated methodology report (build logic/parameter justification/cross-validation), one-click MD + JSON export |

### 3. 61 Built-in Traditional Factors

A ready-to-use factor library covering five categories: Price Trend (18), Volatility (9), Trading Strength (15), Price-Volume (10), and Volume-Derived (9). Each factor includes name, category, tags, formula (Markdown + LaTeX), and reference Pandas implementation code. Supports search by category/keyword and batch export.

### 4. Enhanced Genetic Programming

Introduces three enhancements over traditional GP: **Factor Clusters** (maintain intra-cluster diversity), **Island Model** (multi-population independent evolution with periodic elite migration), and **Event Windows** (market-condition-triggered factor recombination/elimination). Built-in 15 operators (arithmetic, comparison, time-series, cross-sectional ranking).

### 5. Unstructured Data Factor Mining

Extracts Alpha signals from multi-modal text data: `TextAnalyzer` (tokenization, entity recognition, sentiment quantification), `AlternativeDataManager` (supply chain, sentiment, satellite text), and `UnstructuredFactorIntegrator` (fusion with structured factors, incremental information contribution evaluation).

### 6. Transformer-Agent Deep Coupling

Deeply couples Transformer vector representations with the Agent's cognitive loop: `TransformerCoupling`, `CouplingScheduler`, `AgentContextBuilder`, and `AttentionVisualizer` form a "perception → reasoning → action" closed loop, significantly enhancing factor discovery depth and interpretability.

### 7. Local Deployment & Offline Resilience

- **Ollama Integration**: One-click script switches to local LLM (qwen2.5-coder:7b, llama3.1:8b, etc.) — no API key needed
- **Kronos Integration**: Financial time-series forecasting model as predictive factor enhancement
- **Graceful Degradation**: All heavy dependencies (Transformer, RL, ChromaDB) auto-degrade to numpy/heuristic/keyword fallbacks, ensuring zero-dependency-offline operation
- **Preflight Check**: Built-in health check script for conference presentation readiness
- **Data Caching**: Multi-source auto-fallback (EastMoney → Sina → Tushare → THS → Synthetic), with local cache for complete offline operation

### 8. Research Report Knowledge Pipeline (Tencent ima Integration)

FactorGPT keeps its factor knowledge base fresh by continuously watching a live sell-side research library through the Tencent **ima** open API. Three scripts under `scripts/` cover the full loop:

| Script | Role | Cost per run |
|--------|------|--------------|
| `ima_sync.py` | Pulls documents from **your own** ima knowledge base, chunks them, and feeds `data/knowledge/**/chunks.jsonl` + ChromaDB so the Agent can retrieve them | Depends on library size |
| `ima_keyword_watch.py` | **Lightweight watcher (recommended).** Runs `search_knowledge` against a curated keyword list and reports only reports that are new relative to a saved baseline | ~14-28 API calls |
| `ima_subscription_track.py` | **Full directory snapshot.** Walks the entire folder tree of a subscribed library and diffs it against the previous manifest | 350+ paged calls, resumable |

The watcher is the practical entry point. Against a 17,837-document subscribed research library, a full enumeration needs 350+ paged requests and reliably trips the account-level rate limit (`220021`), whereas keyword-targeted search costs roughly one request per keyword and finishes in seconds. Keyword selection matters: precise research terms such as `选股因子` or `量化择时` return a handful to a couple dozen documents, while broad category words such as `ETF` or `期权` return 100+ per page and drown the signal.

```bash
# First run: establish the baseline without reporting everything as "new"
python scripts/ima_keyword_watch.py --init --no-push

# Daily run: report only genuinely new reports, then commit and push the manifest
python scripts/ima_keyword_watch.py

# Adjust the watchlist
python scripts/ima_keyword_watch.py --add-keyword 因子拥挤度
```

Outputs land in `ima_subscription/`: `watch_keywords.json` (watchlist), `keyword_seen.json` (baseline), `keyword_hits.csv` (flat index), and `keyword_watch.md` (append-only log of new arrivals). Both scripts tolerate rate limiting by backing off and checkpointing, and `ima_subscription_track.py` resumes from the last completed folder on the next run instead of restarting the crawl.

Credentials go in `.env` as `IMA_CLIENT_ID` and `IMA_API_KEY` (issued at `ima.qq.com/agent-interface`, valid for one month). Note the API boundary: subscribed/shared libraries allow search but deny full-text export (`get_media_info` returns `220030`), so copying a report into your own library remains a manual step in the ima client — the pipeline reduces that to ticking items off a change list rather than browsing 17k documents.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web UI (17 Pages)            │
├─────────────────────────────────────────────────────────┤
│  Factor Mining Agent (LangGraph)                         │
│  Retrieve → Generate → Validate → Evaluate → Reflect     │
├─────────────────────────────────────────────────────────┤
│  Six-Stage Factor Refinery Pipeline                      │
│  Ore → Mining → Grinding → Screening → Blending → Report │
├──────────────┬──────────────┬────────────────────────────┤
│  LLM Layer   │  Data Layer  │  Engine Layer               │
│  DeepSeek    │  AKShare     │  Sandbox (Subprocess)       │
│  OpenAI      │  Tushare     │  Backtester (IC/Quantile)   │
│  Ollama      │  Sina/THS    │  RPN Engine                 │
│  vLLM        │  Baostock    │  Genetic Programming        │
├──────────────┴──────────────┴────────────────────────────┤
│  Knowledge Base: ChromaDB + BGE Embeddings + 61 Factors   │
│  Experiment Tracking: MLflow / Local JSONL                │
└─────────────────────────────────────────────────────────┘
```

---

## Screenshots & Visualizations

### Backtest Analysis Charts

<p align="center">
  <img src="docs/assets/ic_series.png" alt="IC Time Series" width="48%">
  <img src="docs/assets/quantile_returns.png" alt="Quantile Returns" width="48%">
</p>

<p align="center">
  <img src="docs/assets/quantile_cum.png" alt="Quantile Cumulative Returns" width="48%">
  <img src="docs/assets/long_short.png" alt="Long-Short Cumulative Returns" width="48%">
</p>

<p align="center">
  <img src="docs/assets/portfolio_nav.png" alt="Portfolio NAV" width="48%">
  <img src="docs/assets/factor_ic_bars.png" alt="Multi-Factor IC Comparison" width="48%">
</p>

### Web Interface Screenshots

<p align="center">
  <img src="docs/assets/ui_overview.png" alt="System Overview" width="48%">
  <img src="docs/assets/ui_refinery.png" alt="Factor Refinery" width="48%">
</p>

<p align="center">
  <img src="docs/assets/ui_library.png" alt="Factor Library" width="48%">
  <img src="docs/assets/ui_sysbuild.png" alt="Factor System Builder" width="48%">
</p>

---

## Environment Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11 | 3.12+ |
| RAM | 8 GB | 16 GB |
| Disk Space | 3 GB | 5 GB (with model weights) |
| GPU | Not required | CUDA-compatible GPU for Transformer/RL |
| OS | Linux / macOS / Windows | Ubuntu 22.04+ |

### Optional Dependencies

- **LLM Backend**: Ollama (local), DeepSeek API, OpenAI API, or any OpenAI-compatible endpoint
- **Data Sources**: AKShare (free, no registration), Tushare Pro (free token), Baostock (free)
- **Heavy Modules**: PyTorch (Transformer encoder), stable-baselines3 + sb3-contrib (MaskablePPO), MLflow (experiment tracking) — all auto-degrade if missing

---

## Docker Deployment

### One-Click Start

```bash
# Clone and start
git clone https://github.com/factor-gpt/factor-gpt.git
cd factor-gpt
docker-compose up -d
```

### Build from Source

```bash
docker build -t factorgpt:latest .
docker run -d -p 8501:8501 --name factorgpt factorgpt:latest
```

### Docker Compose Configuration

The included `docker-compose.yml` provides:
- Persistent volume for data cache and ChromaDB
- Environment variable configuration for API keys and data sources
- Automatic port mapping (8501 for Streamlit)
- Resource limits (4GB memory, 2 CPU cores)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `TUSHARE_TOKEN` | Tushare Pro token | - |
| `FACTORGPT_LLM_PROVIDER` | Override LLM provider | ollama |
| `FACTORGPT_LLM_MODEL` | Override LLM model | qwen2.5-coder:7b |
| `HF_ENDPOINT` | HuggingFace mirror endpoint | https://hf-mirror.com |

---

## Sample Data

The repository includes sample data and can generate synthetic data for offline demonstration:

- **Synthetic Data**: Automatically generated with controlled signal-to-noise ratio for testing and CI
- **Real Data**: Use `python scripts/prefetch_data.py` to pull real A-share market data (CSI 800 constituents, daily frequency, 2019-2024)
- **Factor Library**: `data/learned_factors.jsonl` — a growing library of learned and imported factors
- **Vibe-Trading Catalog**: `data/vibe_trading_alpha_catalog.json` — natural language Alpha signal reference catalog
- **Research Watchlist**: `ima_subscription/keyword_hits.csv` — auto-maintained index of matched sell-side research reports

---

## Project Structure

```
factor-gpt/
├── src/
│   ├── agent/          # LangGraph Agent (graph, nodes, state, integration)
│   ├── engine/         # Factor builder, backtester, optimizer, traditional factors
│   ├── data/           # Data fetcher, cleaner, feature forge
│   ├── pipeline/       # Six-stage refinery pipeline
│   ├── rag/            # Knowledge base (ChromaDB + retrieval)
│   ├── llm/            # LLM client (DeepSeek/OpenAI/Ollama compatible)
│   ├── ui/             # Streamlit web interface (17 pages)
│   ├── store/          # SQLite persistence (memory, chat, experiments)
│   └── kronos/         # Kronos financial forecasting model integration
├── scripts/            # Utilities (data prefetch, health check, import, ima sync/watch)
├── docs/assets/        # Documentation screenshots and charts
├── data/               # Sample data, factor library, experiment tracking
├── ima_subscription/   # Research-report watchlist, baseline, and change log
├── config.yaml         # Main configuration file
├── run_agent.py        # CLI entry point
├── Dockerfile          # Docker build file
├── docker-compose.yml  # Docker Compose orchestration
├── requirements.txt    # Python dependencies
└── requirements.lock.txt  # Locked dependencies with hashes (reproducible)
```

---

## Configuration

All settings are centralized in `config.yaml`:

- **llm**: Model provider (deepseek/openai/qwen/ollama), API key, endpoint, temperature, multi-LLM routing
- **data**: Primary data source (akshare/tushare/sina), date range, caching, synthetic fallback
- **backtest**: Quantile count, commission rate, risk-free rate, chart output
- **rag**: Vector store toggle (ChromaDB+BGE or jieba fallback), embedding model, HF mirror
- **agent**: Max iterations, IC threshold, OOS validation
- **refinery**: Six-stage pipeline configuration (Transformer, RL, screening, AlphaPool)
- **proxy**: HTTP/HTTPS proxy for mainland China network environments
- **experiment_tracking**: Experiment logging (local JSONL or MLflow)

---

## Offline & Conference-Ready

FactorGPT is designed for reliable offline demonstrations:

```bash
# Step 1: Prefetch real data (with network)
python scripts/prefetch_data.py

# Step 2: Run health check
python scripts/preflight_check.py --offline

# Step 3: Disconnect network and run
streamlit run src/ui/app.py
python run_agent.py --refinery "Momentum + Quality factors"
```

The system checks five risk categories: RL dependencies, local Ollama models, cached market data, ChromaDB availability, and sandbox stability — all with clear pass/fail/warn outputs.

---

## Roadmap

- [ ] Multi-market support (US stocks, Hong Kong stocks, crypto)
- [ ] Real-time factor monitoring dashboard with alerting
- [ ] Factor decay analysis and lifecycle management
- [ ] Collaborative factor review workflow
- [ ] REST API for programmatic factor mining
- [ ] Integration with backtesting frameworks (Zipline, Backtrader)

---

## Contributing

Contributions are welcome! Please see the issues page for open tasks or submit a pull request with your improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

FactorGPT is an academic research tool for quantitative finance education and research purposes. It does not constitute financial advice. All factor outputs, backtest results, and investment signals are for reference only. Past performance does not guarantee future results. Users should make independent investment decisions based on their own risk tolerance and due diligence.

---

<p align="center">
  <b>FactorGPT</b> — Where Natural Language Meets Quantitative Finance<br>
  <sub>Built with ❤️ for the quantitative finance community</sub>
</p>
