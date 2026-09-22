# FactorGPT — LLM-Powered Quantitative Factor Mining & Industrialization Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)](https://hub.docker.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![HF Spaces](https://img.shields.io/badge/🤗%20Demo-Online-ff9a00?logo=huggingface)](https://huggingface.co/spaces/ZXTLQQ/factorgpt-demo)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/ZXTLQQ/FactorGPT)
[![CI](https://github.com/ZXTLQQ/FactorGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/ZXTLQQ/FactorGPT/actions/workflows/ci.yml)

**FactorGPT** is an LLM-powered intelligent financial factor industrialization platform that deeply integrates natural language understanding with quantitative finance factor engineering. It supports automated factor extraction, validation, combination optimization, and production-grade deployment from both structured and unstructured data sources — all driven by natural language commands.

> **Keywords**: Quantitative Finance, Alpha Factor Mining, LLM Agent, Factor Backtesting, Factor Library, Genetic Programming, Reinforcement Learning, Alternative Data, Streamlit, A-Share, Financial AI, FactorGPT, Factor Refinery, RPN Engine, IC Analysis, Multi-factor Model, LangGraph, Python Quant, EastMoney Miaoxiang MX API, NeoData, Forward Testing, Headline Arena, CRPS, Operator Grid Miner, Factor Expression DSL, PIT Alignment, Factor Crowding, Spectral Cleaning

---

## Table of Contents

- [What Problem Does FactorGPT Solve?](#what-problem-does-factorgpt-solve)
- [Try It Online](#try-it-online)
- [Quick Start](#quick-start)
- [Project Highlights](#project-highlights)
- [Architecture Overview](#architecture-overview)
- [Data Sources & Configuration](#data-sources--configuration)
- [Docker Deployment](#docker-deployment)
- [Screenshots](#screenshots)
- [Environment Requirements](#environment-requirements)
- [Project Structure](#project-structure)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## What Problem Does FactorGPT Solve?

Traditional quantitative factor research faces three major pain points: **high barrier to entry** (requires extensive domain expertise and programming skills), **slow iteration cycles** (manual factor design, coding, and backtesting loops take days to weeks), and **siloed workflows** (data, generation, evaluation, and deployment are disconnected).

FactorGPT addresses these challenges by:

1. **Natural Language to Factor Code**: Describe your investment idea in plain language, and the LLM Agent generates, validates, and backtests factor code automatically. The system includes a built-in knowledge base of 62 traditional factors as reference context.

2. **End-to-End Industrial Pipeline**: The "Six-Stage Factor Refinery" mimics an industrial smelting process — from raw data ore to finished factor products — with built-in quality control at every stage.

3. **Safety and Reliability**: A sandboxed execution environment with white-listed imports, lookahead bias detection (AST-based), automatic winsorization/neutralization/standardization, and out-of-sample validation ensures production-ready factor quality.

4. **Multi-Model Support**: Works with DeepSeek, OpenAI, Qwen, local Ollama models, and any OpenAI-compatible endpoint. The system auto-degrades gracefully when dependencies are missing, ensuring it runs in any environment.

---

## Try It Online

<p align="center">
  <a href="https://huggingface.co/spaces/ZXTLQQ/factorgpt-demo">
    <img src="https://img.shields.io/badge/🤗-Open_in_HuggingFace_Spaces-ff9a00?style=for-the-badge&logo=huggingface" alt="HuggingFace Spaces">
  </a>
</p>

FactorGPT provides a free online demo on HuggingFace Spaces — **no API keys, no sign-up, no installation required**. Describe your investment idea in natural language and see the factor go from code generation to backtest results in seconds. The demo includes: factor mining, 62-factor library browser, interactive IC charts, and a six-stage refinery pipeline walkthrough.

---

## Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **OS**: Linux, macOS, or Windows
- **Memory**: 8 GB RAM minimum (16 GB recommended for Transformer/RL modules)
- **Disk**: ~3 GB core install, ~5 GB if including model weights

### Option 1: Docker One-Click Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/ZXTLQQ/FactorGPT.git
cd FactorGPT

# Build and start
docker-compose up -d

# Open browser at http://localhost:8501
```

The Docker image comes pre-configured with all dependencies, synthetic sample data, and a fully offline-capable demonstration environment. No API keys or network access required. See [Docker Deployment](#docker-deployment) for build-from-source options and environment variables.

### Option 2: Local Installation

```bash
# 1. Clone and enter directory
git clone https://github.com/ZXTLQQ/FactorGPT.git
cd FactorGPT

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

Open your browser at `http://localhost:8501` to access the 21-page integrated web dashboard.

### Credentials: `.env`, never `config.yaml`

`config.yaml` is commit-tracked, so secrets never go into it. Every secret field accepts a `${VAR}` reference that is resolved from the environment at load time — `.env` in the repository root (itself git-ignored) is loaded first, which is what `config.yaml` already ships with:

```yaml
llm:
  api_key: "${DEEPSEEK_API_KEY}"     # resolved from .env / the environment
data:
  tushare_token: "${TUSHARE_TOKEN}"
```

For a local run, copy the keys you actually need from [`.env.example`](.env.example) into `.env`:

```bash
DEEPSEEK_API_KEY=sk-...
TUSHARE_TOKEN=...
```

The sidebar's **保存配置 (Save)** button follows the same rule and tells you so when it runs: the model key and the data-source tokens are written to `.env` (permission-tightened to `0600` on POSIX), `config.yaml` only ever receives the `${VAR}` placeholder, and the value you typed is injected into the running process immediately, so no restart is needed. Saving patches `config.yaml` line by line instead of re-serialising it, so comments, quoting and section order survive — a save that changes nothing leaves the file byte-identical, and switching provider writes to that provider's variable (`DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` / `FACTORGPT_LLM_API_KEY`; Ollama's `ollama` placeholder is not a secret and stays inline).

Two failure modes are deliberately loud rather than silent:

- **Unresolved placeholder.** If `${VAR}` cannot be resolved, the LLM call fails with a message naming the variable instead of sending a fake key for a guaranteed 401. It fails at call time, not at construction time — raising in `__init__` would take the whole Streamlit app down.
- **Offline fallback.** When the model is unreachable the factor pipeline still returns a keyword-template factor, so the run does not die — but the report header is stamped with `因子来源：模板兜底（并非由大模型生成）` plus the reason, and the result badge in the UI says `template`, not `llm`. A template product cannot pass as model output.

Related: **测试连接 (Test connection)** proves that the values *in the input boxes* work; it is not the same as "applied". The panel warns when the boxes and the effective session config diverge, because "it connected but the run used the old settings" is otherwise unexplainable.

### Quick Test (No Network Required)

```bash
# Single factor mining (offline, auto-fallback to synthetic data)
python run_agent.py "Build a 20-day momentum factor"

# Six-stage refinery pipeline (offline demo)
python run_agent.py --refinery "Mix daily and monthly frequency, combine short-term reversal with liquidity"

# Run preflight check (10-second health check for offline presentation readiness)
python scripts/preflight_check.py
```

### One-Command Simulation Demo

```bash
python demo_sim.py
```

Simulates the entire end-to-end factor pipeline on synthetic data — no network or API keys required — and outputs four realistic backtest charts under `demo_output/`: IC time series, quantile returns, long-short cumulative, and quantile cumulative returns.

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

See the backtest IC charts in [Screenshots](#screenshots).

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

### 3. 62 Built-in Traditional Factors

A ready-to-use factor library covering five categories. Each factor includes name, category, tags, formula (Markdown + LaTeX), and reference Pandas implementation code. Supports search by category/keyword and batch export.

![Factor library distribution](docs/assets/feature_factor_library.png)

### 4. Enhanced Genetic Programming

Introduces three enhancements over traditional GP: **Factor Clusters** (maintain intra-cluster diversity), **Island Model** (multi-population independent evolution with periodic elite migration), and **Event Windows** (market-condition-triggered factor recombination/elimination). Built-in 15 operators (arithmetic, comparison, time-series, cross-sectional ranking).

**Where the bars come from is a choice on the page, not a hidden default.** 批量演化 takes a data source (the bundled whole-mine cache `real_ore.pkl`, the offline Parquet set under `data/offline/`, an online fetch, or a synthetic panel), an index pool, a size cap, and an optional comma-separated list of stock codes; `engine.factor_system.resolve_market_panel` is the single entry point behind all four, so the mining script and the page cannot drift apart. Requested codes are normalised (`600519` / `SH600519` / `600519.SH` all resolve to the same name), filtered against the cache, and fetched *directly* online — never "take the whole pool, then truncate", which silently drops exactly the names you asked for. Codes that cannot be resolved come back in the metadata, and if *none* of them resolve the panel is empty and the reason is shown: evolving on placeholder symbols (`S000001`…) and reporting an IC for them would be a result nobody can trace back.

![Enhanced GP evolution](docs/assets/feature_gp_evolution.png)

*Panels: (a) per-cluster convergence, migration generations marked; (b) uniqueness vs. invalid-ratio; (c) real expression tree of the top factor; (d) per-cluster evolution gain; (e) OOS train-vs-test IC; (f) event-window weighted fitness. Rendered from the bundled offline data.

### 5. Unstructured Data Factor Mining

Extracts Alpha signals from multi-modal text data: `TextAnalyzer` (tokenization, entity recognition, sentiment quantification), `AlternativeDataManager` (supply chain, sentiment, satellite text), and `UnstructuredFactorIntegrator` (fusion with structured factors, incremental information contribution evaluation).

![Unstructured text sentiment](docs/assets/feature_unstructured.png)

### 6. Transformer-Agent Deep Coupling

Deeply couples Transformer vector representations with the Agent's cognitive loop: `TransformerCoupling`, `CouplingScheduler`, `AgentContextBuilder`, and `AttentionVisualizer` form a "perception → reasoning → action" closed loop, significantly enhancing factor discovery depth and interpretability.

![Transformer-Agent coupling retrieval](docs/assets/feature_transformer_coupling.png)

### 7. Local Deployment & Offline Resilience

- **Ollama Integration**: One-click script switches to local LLM (qwen2.5-coder:7b, llama3.1:8b, etc.) — no API key needed
- **Kronos Integration**: Financial time-series forecasting model as predictive factor enhancement
- **Graceful Degradation**: All heavy dependencies (Transformer, RL, ChromaDB) auto-degrade to numpy/heuristic/keyword fallbacks, ensuring zero-dependency-offline operation
- **Preflight Check**: Built-in health check script for conference presentation readiness
- **Data Caching**: Multi-source auto-fallback (EastMoney → Sina → Tushare → THS → Synthetic), with local cache for complete offline operation

See [Offline Data Source](#offline-data-source-built-in-no-network) for the bundled offline dataset.

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

![ima keyword hits](docs/assets/feature_ima_pipeline.png)

### 9. Forward-Testing via Headline Arena (前瞻检验，独立于回测)

Factor validation (IC/IR backtests) is inherently **historical** — no matter how rigorous, it cannot answer "will this factor view still hold going forward?" FactorGPT closes that gap with a third-party, pre-committed forward-testing line through **Headline Arena** (`headlinearena.com`): the macro views embedded in factor-layer research (rates, style, commodity direction) are converted into daily probability predictions on global macro assets (gold `GC`, 10Y treasury `ZN`, WTI crude `CL`, E-mini S&P 500 `ES`, silver `SI`, copper `HG`, dollar `DXY`, …). Each prediction's settlement standard is **frozen at question creation**, predictions are **locked before the outcome exists**, and settlement is **mechanically scored by a third party** against real market data (directional: `50 ± confidence×50`; the platform also publishes per-agent CRPS/Brier calibration APIs). That makes the forward line the hardest-to-dispute form of evidence for factor validity.

```bash
# 1) One-time: register an HA agent (client_secret shown once, saved to ~/.headlinearena;
#    then a human claims the agent via claim_url + pairing_code)
python scripts/ha_forward_run.py register --name FactorGPT-GoldBot --bio "Gold macro view forward tests" \
    --model-provider DeepSeek --model-name deepseek-chat

# 2) Daily forward loop — turn a macro theme (or a factor methodology report) into locked predictions.
#    Default is dry_run: predictions are locked into the local ledger, no network/credentials required.
python scripts/ha_forward_run.py run --theme "避险情绪升温，降息预期增强，油价上行"
python scripts/ha_forward_run.py run --factor-report output/methodology_report.json
python scripts/ha_forward_run.py status                 # ledger state
python scripts/ha_forward_run.py scorecard              # accuracy / Brier / calibration card (Markdown)

# 3) Live submission (needs credentials + claimed agent; auto-subscribes asset scopes)
python scripts/ha_forward_run.py run --live --theme "油价上行，供给收紧" --assets GC,CL

# 4) Settlement backfill: pull third-party results into the ledger, then re-run scorecard
python scripts/ha_forward_run.py settle
```

Mechanics and guarantees: the bridge lives in `src/forwardtest/` (pure stdlib, zero new dependencies): `client.py` (register/auth/scope/challenges/predict/results/calibration), `translator.py` (deterministic macro-theme → asset/direction/confidence mapping), `ledger.py` (append-only JSONL under `data/forwardtest/`; prediction fields are frozen once settled), `scorecard.py` (directional accuracy, Brier vs the 1/3 random baseline, confidence-bucket calibration), and `runner.py` (orchestration). Every run degrades gracefully — no network, no credentials, or a factor description without macro wording simply skips or writes a local dry-run record, never an exception that breaks the factor pipeline. Optional integration: with `headline_arena.enabled: true` in `config.yaml`, the LangGraph agent appends a **shadow forward-test node** after `finalize`, folding each mined factor's description into a locked dry-run ledger record (agents never submit live predictions on their own). Credentials go in `.env` as `HA_AGENT_ID` / `HA_CLIENT_SECRET` (or `~/.headlinearena/credentials.json`); the platform is free, prediction rewards convert to LLM-gateway credits, and the plugin/API references are `github.com/headlinearena/headlinearena-agent-plugin` and `headlinearena.com/api/v1/agent/onboarding/guide.txt`.

### 10. Typed Factor Research Layer (`src/mining/`)

`src/mining/` is a self-contained factor-research layer (~4,300 lines) that turns a research workflow into a typed expression language, a PIT-safe panel, reproducible factor libraries, and a four-dimension evaluation that collapses to a single comparable score — fully offline, no LLM in the loop.

| Layer | What landed |
|-------|-------------|
| Typed expression DSL, operator library, grid miner, evaluator & run report | 60 registered operators in six families (`elem` 10 / `elem2` 11 / `cs` 6 / `ts` 27 / `ts2` 5 / `cs2` 1); a **typed expression DSL** (`parse` / `validate` / `infer_type` / `render`, with commutativity-aware key de-duplication) whose dimension gate rejects unit-incoherent arithmetic (price + turnover, flags in `log`, …) at build time; an **operator grid miner** that searches operator × window × layer combinations under a hard evaluation budget; `evaluator.py`, which splits "is this factor good?" into **four dimensions** — data quality / predictive power / stability / correlation — and folds them into one weighted score *after* subtracting turnover-cost and expression-complexity penalties, because searching on IC alone is guaranteed to overfit; and `report.py`, which renders the whole run — layer-by-layer prune counts, per-factor scores, risk gate, correlations, incremental IC and the exact config — into a self-contained Markdown + JSON report |
| Risk & crowding gate | `risk.py`: exposure regression ΔR², Newey-West-adjusted t-stat, VIF, lag-1 autocorrelation, crowding score, and component risk contribution (components sum to portfolio vol) |
| PIT fundamental layer | `fundamental.py`: PIT installation of quarterly financials; TTM expressed purely with existing operators (`add(add(x, ts_delay(x, 250)), add(ts_delay(x, 500), ts_delay(x, 750)))` = sum of four single-quarter values); 31 fundamental factors (quality / growth / leverage / accrual / valuation) plus cross-domain combination templates |
| Concept layer & market-cap grouping | `concept.py`: interval-valid concept membership (differential counting, **no forward-fill**), concept count / niche / heat fields, 20 concept factors, and the **DGTW market-cap grouping operator** (`dgtw_cs`) that absorbs the non-linear part of the size relation which linear neutralisation leaves behind |

Anti-lookahead is enforced structurally rather than by convention:

- **`asof_align`** projects `announcement date + lag` onto trading days, and **`pit_reference`** re-derives the same series with a deliberately naive loop as an independent cross-check; `check_pit` compares them point by point *and* separately asserts "no value before the first visible day", so a shared bug cannot hide behind itself.
- **`lookback`** computes the mandatory warm-up window of any expression (`ts_mean(ts_delay(close, 250), 20)` → 269 days) and **`coverage_adj`** measures coverage *after* that warm-up — so "not enough history" is no longer misdiagnosed as "dead factor" (`check_history` fails loudly instead).
- **RankIC** ranks each variable cross-sectionally first, then correlates on the pairwise-complete sample (the pandas `rank → corrwith` convention). This definition is pinned by `tests/test_mining.py::test_cs_corr_rank_convention` so it cannot be silently "fixed" later.

```bash
python -m pytest tests/test_mining.py -q            # 32 tests, ~15 s, fully offline
python -m pytest tests/test_triage.py -q            # 26 tests — the acceptance gate (see below)
python scripts/mining_report_demo.py                # one-command demo → demo_output/mining_report.md
```

The demo builds a synthetic panel, installs PIT fundamentals and concept memberships, runs a grid search, evaluates four representative factors from the three factor libraries, runs the acceptance gate (statistical significance / universe domains / multi-scale mining — `--no-acceptance` skips it), and writes `demo_output/mining_report.md` (+`.json`). No network, no API keys — the missing-value convention in the report is a hard rule: unmeasurable cells render as `—` and the JSON payload is written with `allow_nan=False`, so a `nan` can never masquerade as a real number.

### 11. AI Factor-System Advisor (`src/engine/system_advisor.py`)

A single backtest produces dozens of numbers, but the decisions a researcher actually has to make are only a handful: can this system enter the portfolio, which factors double-count the same exposure, which one carries the risk, and what to change first. `src/engine/system_advisor.py` compresses one backtest result into a single **fact table** (`distill`) and lets two answer paths share it:

- **Rule path** (`build_rule_answer`) — pure local, zero external calls. Eleven intents (overall quality / overfitting / spectral noise / redundancy / weighting / risk concentration / decay / add-or-drop / capacity / next actions / fallback) each translate the fact table into a verdict plus prioritized actions. It guarantees an actionable answer even with no model configured, and it doubles as the numeric baseline for the LLM path.
- **LLM path** (`advise(question, result, llm=...)`) — the fact table is the *only* injected context, and the system prompt forbids citing any number that is not in it, so the model cannot dress up a plausible-but-invented figure as analysis. A model failure never raises: the answer degrades to the rule path and the reason is returned in `error`, because a consulting window must not go blank when the provider is down.

Both paths read the same facts, so a new metric has to be added in exactly one place — the "the dashboard shows this number but the advisor cannot cite it" class of drift is structurally impossible. Thresholds are shared with `factor_system.build_findings` (IC ≥ 0.03 and ICIR ≥ 0.4 = effective, |ρ| ≥ 0.8 = redundant, single-factor risk share ≥ 40% = concentrated); otherwise the same backtest would be judged twice, with two different verdicts. The chat page (`因子体系 → AI 体系咨询`) keeps per-system conversation history in the same SQLite store, labels every answer with its source (LLM vs. rule engine) and detected intent, and exposes the fact table itself — so any answer can be audited against its inputs.

### 12. Spectral Cleaning & Risk Decomposition for Factor Systems (`src/engine/eigen_clean.py`)

A factor system can look diversified and still be a single bet. The eigenvalues of a sample correlation matrix mix real signal directions with directions that are pure estimation noise, and a weight optimiser handed that matrix will load up on directions that are "independent" only in-sample: in-sample variance looks low, and part of that diversification is a gift from estimation error. The chain lives in `src/engine/eigen_clean.py`, split so every step can be checked on its own:

| Step | Entry point | Question it answers |
|------|-------------|---------------------|
| Noise boundary | `mp_bounds` | which eigenvalues are indistinguishable from random-matrix noise (λ₊ / λ₋) |
| Spectrum cleaning | `analyze_spectrum` | clean by lifting noise eigenvalues to the noise mean, or by shrinking linearly toward the identity |
| Weight solving | `optimize_weights` | minimum-variance / maximum-diversification / risk-parity / ICIR-tilt weights **on the cleaned matrix**, under `Σw = 1, w ≥ 0` (the output is a stock-ranking score, not a tradable long-short book) |
| Risk decomposition | `marginal_contributions`, `addition_ranking` | how much of system variance each factor owns, and whether adding one more factor is worth it |

`factor_system.analyze_system(..., run_spectral=True)` runs it end-to-end. With fewer than two factors — or with the switch off — the result carries `ok: false` *and the reason*, rather than raising inside a page render. The returned block holds the noise-band boundary, counts of signal / noise / **in-band** directions, the effective number of independent directions before and after cleaning, per-factor marginal risk contributions, and the same weights evaluated on the raw **and** the cleaned matrix: a positive gap there means the raw matrix *understated* system volatility, i.e. some of the in-sample diversification was free. Findings follow the same discipline — in-band directions are reported as **statistically undecidable** instead of being promoted to signal, and cleaning only moves what is *clearly* noise, because a matrix that "improves" everything is a matrix nobody can audit. The page renders it under 因子体系 → 「谱清洗与风险分解」, and the AI advisor (Highlight 11) reads exactly these numbers: its noise-ratio KPI and its spectral-noise intent come from this block, so the chat window cannot cite a divergence that the diagnosis page does not show.

**Engine research layer.** The same revision adds six engine modules: `ic_utils.py` (one vectorised cross-sectional IC kernel — group-wise correlation, panel IC, block and period helpers — shared by the modules below, with non-finite values *excluded* rather than read as 0, because treating a missing value as neutral is the most expensive way to be wrong), `significance.py` (stationary-block bootstrap p-values, the IC that a search over *m* trials must beat to count as more than luck, Benjamini–Hochberg false-discovery control across the candidate table, effective sample size from AR(1) — the gate a genetic search most needs, since looking at the best of several thousand expressions and reporting `ICIR·√n` leaves the "how many did I try?" term out of the degrees of freedom), `universe.py` (per-day tradability labels with the reason attached, three preset liquidity tiers, side-by-side comparison of one factor across domains), `param_ops.py` (a parameterised temporal-memory operator: hyperbolic lag weighting plus tanh saturation, with the structure evolved and its five parameters fitted inside each window), `multiscale_gp.py` (hierarchical multi-scale mining — evolve on a coarse evaluation grid, diagnose the distorted intervals by exposure drift and profile deviation, refine only those, and stitch the two scales together with the coarse value as a terminal cost; its resource report states the evaluation-count saving and the smaller wall-clock saving separately instead of quoting the first as the second) and `specification_rl.py` (multitask reinforcement learning for factor-specification search — see Highlight 13 below).

Four of the six are now **wired end-to-end** rather than library-level. `src/mining/triage.py` bridges the wide panel (dates × symbols) into the long `(date, symbol)` table the engines consume, and a single `acceptance(...)` entry point runs the significance gate, the domain comparison, the multi-scale search and the RL specification search together. The demo (`scripts/mining_report_demo.py`, step 5 of 6) and the mining page (因子挖掘 → 「统计显著性检验」/「选股域对照」/「分层多尺度挖掘」/「规范搜索」) call exactly that entry point, and each gate becomes its own section of the report — a gate that was not given its inputs is *dropped* rather than rendered as an empty table.

The multi-scale search also supports **multi-fold refinement** (`n_folds >= 2`, after the paper's §4.1.2 two-fold experiment): inside each *already selected* interval, the same two diagnostics (exposure drift, profile deviation) run once more on sub-intervals, and only the worst-scoring sub-interval receives the third-scale budget — a candidate is replaced only when the refined expression actually raises its sub-interval IC, so refinement cannot make things worse, and the extra evaluations show up honestly in `ResourceReport.fold2_evals` with the brute-force baseline raised to match.

Two rendering/performance fixes ride along with this revision. First, the mining report's 「四、回测图表」 section embedded *local absolute paths* of the saved PNGs in its markdown; a browser cannot read `file:///e:/...`, so every chart rendered as a broken image — the UI now embeds existing local chart files as base64 data URIs at render time (missing files stay broken *and visible*, rather than being silently hidden), and the agent's `chart_paths` — previously looked up under a key that never existed — render through `st.image`. Second, `FactorBacktester.evaluate` spent ~90% of its wall clock in two per-day Python loops (the daily-IC series re-scanned the whole panel once per day, and quantile grouping called `pd.qcut` once per day through `transform`); both are now single vectorised passes (bincount-aggregated daily correlation, rank-based bucketing), verified equivalent to the per-day references to 1e-12 — on a 300-symbol × 500-day panel `evaluate` drops from 2.21 s to 0.245 s.

Three conventions are load-bearing, and `tests/test_triage.py` (26 tests) pins them:

- **The multiplicity denominator is the number of expressions *this* search evaluated** (`search.n_evaluated`), not the number of candidates. A single-trial threshold applied to the winner of several hundred tries is the same as deleting the "how many did I try?" term from the degrees of freedom.
- **The universe tiers are cut on the panel's own turnover quantiles** (`min_history` 20 / 60 / 120), not the engine's absolute thresholds (5e6 / 2e7 / 1e8 CNY). On a synthetic or thin panel the absolute strict tier starves to zero tradable rows, which then reads as "the factor dies in liquid names" instead of "my threshold was wrong".
- **The last interval is never selectable** in multi-scale mining: there is no next interval to compare against, so its exposure drift is *undefined* (`—` in the report, `null` in the JSON) rather than zero, and the evaluation-count saving is reported separately from the smaller wall-clock saving.

Wiring the modules up also surfaced two defects that library-only status had hidden, both fixed here: `genetic_enhanced.eval_expr` computed `ts_corr` through `groupby(...).apply`, which hands the grouping column into the callback and warns on pandas ≥ 2.2 — a hard failure under `filterwarnings = error`; it is now a per-symbol rolling correlation written back by position, verified numerically identical to the old formula on 20 random expressions. And `multiscale_gp._py` converted only *numpy* floats, so a plain `float('nan')` reached the payload as a bare `NaN` — not legal JSON, so the UI consumer would have failed to parse the report it had just written.

### 13. Multitask RL for Factor-Specification Search (`src/engine/specification_rl.py`)

Genetic programming searches *expressions*; a researcher actually chooses a *specification* — which modelling terms go in, in what structure, over what window. This module follows Delphos (*Multitask Reinforcement Learning for Assisting Choice Model Specification*, arXiv:2609.18441v1) in casting that choice as an MDP and learning a policy **shared across tasks**, where one task is one panel (or one time slice of a panel):

| MDP element | Here |
|-------------|------|
| State | the set of chosen modelling terms — a *set*, aggregated by mean pooling, so it is **permutation-invariant and independent of variable names**; this is what lets a policy trained on task A transfer to task B whose columns are named differently |
| Action | `add` a term / `change` a term slot / `terminate`, with three **action masks**: components unavailable to the task, terms that would immediately revert, and terms already chosen |
| Reward | `tanh((fit − fit₀) / scale)` on termination, `−1` for a specification that cannot be estimated; `fit = |IC|` because a factor's sign is free |
| Policy | one linear **DeepSet Q-network** shared by all tasks, ϵ-greedy decay, target network `θ⁻`, a shared replay buffer sampled with **balanced mini-batches** (equal contribution per task, so a task with many cheap episodes cannot dominate the gradient) |

The paper's headline claim is zero-shot transfer to unseen datasets, and we reproduce it in the only setting where that claim is falsifiable — `tests/test_specification_rl.py::test_multitask_transfers_to_unseen_features_better_than_single_task` builds synthetic tasks that share the same *optimal modelling concept* while carrying it on **different features**, then evaluates on held-out tasks whose features were never seen in training. At 30 episodes per task the shared policy reaches **0.75** success rate on held-out tasks, single-task training reaches **0.19**, and an untrained random policy **0.32** — the shared policy wins on 8/8 seeds, and single-task training scores *below random* because it learns feature-specific preferences that do not carry.

Three deliberate departures from the paper are documented in the module docstring with the measurements that forced them: a **linear** Q over DeepSet features (a deep net is unjustifiable at this state dimensionality and would make the transfer test uninterpretable), **sampled** actions (the default catalogue admits ~2·10⁴ actions; `sample_actions` keeps the mask semantics while cutting an episode to milliseconds), and **reward-to-go instead of TD(0)** — with short trajectories and a `max` over a linear Q, one-step bootstrapping propagates almost no signal and the learning curve went *down*; bootstrapping truncated episodes is implemented behind `bootstrap_truncated` and left **off by default** because turning it on drops held-out success from 0.75 to 0.49.

It is wired in where it can change an outcome, not only a number: `triage.specification_search(...)` / `acceptance(...)` run it, the mining page renders it under 因子挖掘 → 「🧠 规范搜索」 (per-task candidate table with fit, |IC|, complexity and Pareto membership), the report adds a 「多任务规范搜索（强化学习）」 section, and — most usefully — `multiscale_mine(..., spec_rl=True)` feeds `spec_seeds()` into `HierarchicalFactorMiner(seed_exprs=...)`: the RL proposals are compiled through `compile_expr` into the *same* expression-tree tuples the GP evolves, so the policy seeds the genetic initial population instead of sitting beside it.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Web UI (21 Pages)                  │
├─────────────────────────────────────────────────────────────┤
│              Factor Mining Agent (LangGraph)                  │
│        Retrieve → Generate → Validate → Evaluate → Reflect    │
├─────────────────────────────────────────────────────────────┤
│              Six-Stage Factor Refinery Pipeline               │
│       Ore → Mining → Grinding → Screening → Blending → Report │
├───────────────┬──────────────────┬───────────────────────────┤
│   LLM Layer   │    Data Layer    │       Engine Layer         │
│   DeepSeek    │    AKShare       │   Sandbox (Subprocess)     │
│   OpenAI      │    Tushare       │   Backtester (IC/Quantile) │
│   Ollama      │    Sina / THS    │   RPN Engine               │
│   vLLM        │    Baostock      │   Genetic Programming      │
│               │    MX Miaoxiang  │   Transformer / RL         │
│               │    NeoData       │                            │
├───────────────┴──────────────────┴───────────────────────────┤
│   Knowledge Base: ChromaDB + BGE Embeddings + 62 Factors     │
│   Experiment Tracking: MLflow / Local JSONL                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Sources & Configuration

### Configuration

All settings are centralized in `config.yaml`:

- **llm**: Model provider (deepseek/openai/qwen/ollama), API key, endpoint, temperature, multi-LLM routing. The key may be written as `${VAR}` and resolved from `.env` — see [Credentials](#credentials-env-never-configyaml); the sidebar's save button always writes it that way.
- **data**: Primary data source, date range, caching, synthetic fallback. `data.source` accepts three values:
  - `offline` (**default**) — built-in bundled local dataset, no network required
  - `legacy` — self-crawled akshare/sina/tushare sources
  - `neodata` — platform stable data source (experimental, see below)
- **backtest**: Quantile count, commission rate, risk-free rate, chart output
- **rag**: Vector store toggle (ChromaDB+BGE or jieba fallback), embedding model, HF mirror
- **agent**: Max iterations, IC threshold, OOS validation
- **refinery**: Six-stage pipeline configuration (Transformer, RL, screening, AlphaPool)
- **proxy**: HTTP/HTTPS proxy for mainland China network environments
- **experiment_tracking**: Experiment logging (local JSONL or MLflow)
- **headline_arena**: Forward-testing of factor-layer macro views via Headline Arena (enabled toggles the LangGraph shadow node; `dry_run: true` keeps predictions local by default)

Secret fields (`llm.api_key`, `data.tushare_token`, `data.ths_api_token`) are never stored literally: `config.yaml` keeps the `${VAR}` placeholder and the value lives in `.env`. Saving from the UI is a line-level patch, not a re-serialisation, so the comments and section order in `config.yaml` — and the diff you have to review — stay readable.

### NeoData Stable Data Source (Experimental)

FactorGPT can optionally route all market-data calls through the platform's **NeoData** service instead of self-crawling akshare/sina/tushare. The switch is unified behind a factory in `src/data/neo_adapter.py` (`get_data_source()`), so the four call sites (`graph.py`, `refinery.py`, `factor_system.py`, `market_data.py`) are unchanged and the **local `legacy` scheme is fully preserved** by default.

- **How to enable**: set `data.source: neodata` in `config.yaml`. The real gateway `base_url` is already filled in `config.yaml` (`data.neodata.base_url`).
- **Authentication**: requires the platform-scoped `tempToken`, which the platform writes to `~/.workbuddy/.neodata_token` (or the `NEODATA_TOKEN` env var). An ordinary IDE session token will be rejected with HTTP 401.

> **Important limitation — `fallback_to_legacy` must stay `true`.** NeoData is a **natural-language query** service: it returns free-text answer blocks (`data.apiData.apiRecall[].content`), **not** a structured bulk-data API. It therefore cannot reliably provide the structured datasets the factor engine needs — full daily-K-line time series (backtest core), complete index-constituent lists, industry/market-cap mappings, and structured financial statements. The adapter's `neo()` parsers are best-effort and return empty for these, so `fallback_to_legacy` is required to keep backtests runnable. In practice `neodata` currently serves only as a research-Q&A aid and **does not replace `legacy` for factor backtesting**. Live field validation was also blocked in this environment because the platform `tempToken` was not available (session token → 401). Revisit turning `fallback_to_legacy` off only after a valid `tempToken` is obtainable and structured parsing is proven.

### Offline Data Source (built-in, no network)

For a **fully offline** environment (no internet, flaky akshare/sina feeds, or deterministic backtesting), FactorGPT ships with a built-in local market dataset under `data/offline/` — cloned straight from the repository, no setup required. It is read through the `OfflineDataSource` adapter (`src/data/offline_adapter.py`), behind the same `get_data_source()` factory as `legacy`/`neodata` — so all four call sites (`graph.py`, `refinery.py`, `factor_system.py`, `market_data.py`) work unchanged.

- **How to enable**: set `data.source: offline` in `config.yaml` (default).
- **Data files** (bundled, commit-tracked): `data/offline/bars_<index>_part*.parquet` (daily bars, sharded so each file stays under 100 MB), `constituents_<pool>.json` (`csi300` / `csi500` / `csi800` / `csi1000` / `csiall`), `index_daily.parquet` (major broad-market index daily bars, incl. the CSI 800 benchmark `000906`), `trade_calendar.json`, `micro_snapshot.parquet` (micro cross-section: industry / board / area / market cap / valuation, one row per symbol), and `meta.json` (trade range, symbol/row counts, index/calendar/micro coverage). The default pool is `csi800` (1977 symbols, 2019-01 ~ 2026-08, ~3.43M rows). After cloning, the dataset is ready to use — no download, no API key.
- **What it provides**: daily K-line (qfq-adjusted), pct_chg — aligned with the `DataFetcher` column contract (`date/symbol/open/high/low/close/volume/amount/pct_chg`) — plus multi-pool constituents (`get_index_constituents("000300"|"csi500"|"csi800"|...)`), benchmark index daily bars (`get_index_daily("000906")`) and the trading calendar (`get_trade_calendar()`), all three of which are new supplement files built by `scripts/build_offline_dataset.py` from the local Qlib `.bin` store (no `qlib` package needed).
- **Micro cross-section**: `micro_snapshot.parquet` (1977 rows, one per shipped symbol) powers `get_industry_and_cap(symbols, level=1|2|3)` — the same `(industry, mkt_cap)` contract as `DataFetcher`, so industry/market-cap neutralization now also works offline — plus `get_industry_classification(level=)` (per-industry count / cap / median PE-PB), `get_micro_snapshot(symbols)` and `get_market_snapshot(symbols)`. Columns: `symbol / instrument / name / board / industry / industry_l2 / industry_l3 / area / price / total_mv / float_mv / shares_total / shares_float / pe / pb / source / quote_source / as_of`. Coverage: 1977/1977 symbols with industry (27 level-1, 102 level-2 sectors, 33 provinces), 1975/1977 with market cap. `board` (沪市主板/深市主板/创业板/科创板/北交所) is derived locally from the code prefix — no network. Market cap / valuation are a **build-time static snapshot** (`as_of` column), not a live feed, so cap-neutralization is approximate on older dates.
- **Supplement regeneration**: `python scripts/build_offline_dataset.py --qlib-dir E:/Qlib/data/cn_data` — rewrites `index_daily.parquet` / `trade_calendar.json` / `constituents_<pool>.json` and refreshes `meta.json`. Constituents are intersected with the shipped K-line coverage, and indices with <95% trading-day coverage in the range are dropped (e.g. `SH000985`, incomplete in the source) so offline benchmark curves never contain gaps.
- **Micro regeneration**: `python scripts/build_offline_micro.py` (~45 requests: 5 paged EastMoney company-profile calls for industry/area + 40 batched Tencent quote calls for price/market cap/PE/PB; `--universe csi300` narrows the pool, `--as-of` stamps the snapshot date). EastMoney's rate-limited `push2` host is deliberately avoided — the profile report lives on `datacenter-web`, and Sina industry boards serve only as a fallback channel.
- **Physical vs contract schema**: the bundled parquet files store the code column physically as `instrument` (e.g. `sh.600000`); `offline_adapter.py` bridges it to the downstream `symbol` contract via `_de_norm_symbol()` (`instrument→symbol` rename). Never consume `instrument` directly outside the adapter — the documented data contract for all factor/backtest code is `symbol`.
- **What it does not provide**: financial statements, news/sentiment, minute bars, and concept-board / listing-date fields (concept memberships need EastMoney's rate-limited `push2` host). Those dimensions still degrade gracefully to empty, and the factor pipeline keeps running on pure price-volume data plus the benchmark index series.
- **UI**: the sidebar "数据源设置" panel has an `offline` option plus a live status readout (trade range, symbol/row counts from `meta.json`).

![Offline data source coverage](docs/assets/feature_offline_data.png)

### High-Frequency Data Source (offline futures L2 order book, `data.source: hf`)

A second offline source for **high-frequency work**: a full-market futures **L2 5-level snapshot** feed (500 ms polling, ~15.8M rows / 795 contracts / 82 products for a single day) plus your own **order blotter**, read through `HFDataSource` (`src/data/hf_adapter.py`) behind the same `get_data_source()` factory. Set `data.source: hf` to make it serve daily K-line like `offline`, and call the extra methods for order-book work:

```yaml
data:
  source: hf
  hf:
    file: "C:/Users/HP/Desktop/20251208.pqt"      # L2 snapshot parquet
    cache_dir: "data/hf/cache"                     # per-contract slice cache
    orders_file: "C:/Users/HP/Desktop/附件3：20251208_orders.xlsx"
    search_dirs: ["data/hf", "C:/Users/HP/Desktop"]
    glob: "*2025*.pqt"
```

- **Beyond K-line**: `load_l2(contract, sessions=("M","E"))`, `load_symbol_l2(product)`, `get_minute_kline(freq=)`, `get_term_structure(product)`, `load_orders()` and `overlap_diagnostics()` (order-vs-quote second-level alignment check).
- **Mining layer** (`src/mining/hf*.py`): ~50 order-book factors (OFI via price-matched queue change, micro-price, depth imbalance, realized vol, …), forward labels cut at session boundaries, direction model, fill-probability model (AUC 0.795, top-decile fill rate 11.5% vs 2.8% base) and the four strategy families — passive market making, short-term trend, event-driven, calendar (cross-month) arbitrage — all scored with explicit tick costs.
- **Integrated into the typed research layer**: `register_hf_fields()` / `install_hf_features()` put these fields into the same `FieldRegistry` / `PanelData` / expression tree, tagged `ROLE_ALT` (pre-trade alternative info) so they can be combined with price-volume factors and still be caught by dimension checks. Demo: `python scripts/hf_mining_demo.py --freq 1min --symbol au`, `python scripts/hf_strategy_demo.py`.
- **Read `docs/高频数据接入与因子挖掘.md` first.** It documents three traps that silently corrupt results (cross-midnight `SortTime`, session gaps 4 orders of magnitude larger than the sampling interval, and derived columns that are constant zero), plus honest negative results: book factors aggregated to ≥1 min bars carry RankIC of only 0.01–0.05, one order of magnitude below price momentum on the same data — so this layer belongs on the tick grid and must never be judged without transaction costs.

### EastMoney MX (妙想) Data Interface

An official supplement to the NeoData channel: the EastMoney "Miaoxiang" (妙想) open API provides six data capabilities — market/fundamental queries (`data`), news & research search (`search`), smart stock screening (`xuangu`), watchlist management (`zixuan`), simulated portfolio (`moni`), and financial community content (`poster`). It is a reliable replacement for fragile self-crawled akshare/sina feeds.

- **Skill packages**: bundled at `factorgpt-skill/skills/mx-*/` (official releases, one `SKILL.md` + script per package). API reference: https://marketing.dfcfw.com/res/download/A620260623NIYC2U.md
- **API key (never commit it)**: set `MX_APIKEY` in your local `.env` (git-ignored) or as a persistent env var (`setx MX_APIKEY "..."` on Windows). The committed `.env.example` keeps `MX_APIKEY=` empty for users to fill in themselves.
- **Usage**: the cross-platform wrapper `scripts/mx_query.py` injects the key automatically and writes outputs to `output/mx_data/`:

```bash
python scripts/mx_query.py data "上证指数今日行情"
python scripts/mx_query.py search "白酒板块研报"
python scripts/mx_query.py xuangu "市盈率低于10的银行股"
python scripts/mx_query.py --list
```

> The official scripts default to a Linux output path (`/root/.openclaw/workspace/mx_data/output/`); on Windows either pass an explicit output dir or use the `mx_query.py` wrapper.

### Offline & Conference-Ready

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

## Docker Deployment

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
- Resource limits (4 GB memory, 2 CPU cores) to keep demo environments light — raise the memory cap for Transformer/RL workloads

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key (`llm.provider: deepseek`) | - |
| `OPENAI_API_KEY` | OpenAI API key (`llm.provider: openai`) | - |
| `DASHSCOPE_API_KEY` | Qwen / DashScope API key (`llm.provider: qwen`) | - |
| `FACTORGPT_LLM_API_KEY` | API key for any other OpenAI-compatible endpoint (`custom`, `vllm`, OpenRouter, …) | - |
| `TUSHARE_TOKEN` | Tushare Pro token (`data.tushare_token`) | - |
| `THS_API_TOKEN` | Tonghuashun iFinD MCP gateway token (`data.ths_api_token`) | - |
| `FACTORGPT_LLM_PROVIDER` | Override LLM provider | ollama |
| `FACTORGPT_LLM_MODEL` | Override LLM model | qwen2.5-coder:7b |
| `HF_ENDPOINT` | HuggingFace mirror endpoint | https://hf-mirror.com |
| `MX_APIKEY` | EastMoney Miaoxiang (妙想) API key, see "EastMoney MX" section | - |
| `IMA_CLIENT_ID` | Tencent ima client ID for the research-report pipeline | - |
| `IMA_API_KEY` | Tencent ima API key (renew monthly at ima.qq.com/agent-interface) | - |
| `HA_AGENT_ID` | Headline Arena agent id (forward-testing bridge; also saved to ~/.headlinearena/credentials.json) | - |
| `HA_CLIENT_SECRET` | Headline Arena client secret (shown once at registration) | - |

These are exactly the names the sidebar's **保存配置** button writes into `.env`; the first six are the ones `config.yaml` refers to as `${VAR}`. See [Credentials](#credentials-env-never-configyaml) for what happens when one of them is missing.

---

## Screenshots

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
- **Data Sources**: AKShare (free, no registration), Tushare Pro (free token), Baostock (free), EastMoney MX Miaoxiang API (key required), NeoData (platform service)
- **Heavy Modules**: PyTorch (Transformer encoder), stable-baselines3 + sb3-contrib (MaskablePPO), MLflow (experiment tracking) — all auto-degrade if missing

---

## Project Structure

```
FactorGPT/
├── src/
│   ├── agent/          # LangGraph Agent (graph, nodes, state, integration)
│   ├── engine/         # Factor builder, backtester, optimizer, traditional factors, spectral cleaning / significance / universe / multiscale GP / multitask specification RL
│   ├── data/           # Data fetcher, cleaner, feature forge, offline/neo adapters
│   ├── pipeline/       # Six-stage refinery pipeline
│   ├── mining/         # Typed factor research layer (DSL, PIT panel, ops grid miner)
│   ├── rag/            # Knowledge base (ChromaDB + retrieval)
│   ├── llm/            # LLM client (DeepSeek/OpenAI/Ollama compatible)
│   ├── ui/             # Streamlit web interface (21 pages)
│   ├── store/          # SQLite persistence (memory, chat, experiments)
│   ├── forwardtest/    # Headline Arena forward-testing bridge (client/ledger/scorecard)
│   └── kronos/         # Kronos financial forecasting model integration
├── scripts/            # Utilities (data prefetch, health check, mx_query, ima sync/watch, ha_forward_run)
├── tests/              # Test suite (sandbox & lookahead, refinery, mining, forward test, advisor, specification RL, docs contract)
├── factorgpt-skill/    # Agent skill packages (SKILL.md + official EastMoney MX skills)
│   ├── skills/         # mx-data / mx-search / mx-xuangu / mx-zixuan / mx-moni / mx-poster
│   └── references/     # Data contract (legacy / NeoData / offline field mapping)
├── third_party/        # Third-party integrations (kronos, ima client)
├── hf_space/           # HuggingFace Spaces static hosting files
├── .github/workflows/  # CI (pytest on Python 3.11 + 3.12, then compile check)
├── docs/               # Ablation report + docs/assets screenshots and charts
├── data/               # Sample data, factor library, bundled offline dataset, forward-test ledger
├── ima_subscription/   # Research-report watchlist, baseline, and change log
├── demo_output/        # One-command demo backtest charts (python demo_sim.py)
├── config.yaml         # Main configuration file
├── run_agent.py        # CLI entry point
├── demo_sim.py         # One-command end-to-end demo simulation
├── Dockerfile          # Docker build file
├── docker-compose.yml  # Docker Compose orchestration
├── requirements.txt    # Python dependencies
└── requirements.lock.txt  # Locked dependencies with hashes (reproducible)
```

---

## Testing & Quality Assurance

FactorGPT's "production-grade" claim is backed by automated tests and reproducible experiments, not just a badge — 232 test functions across 14 files (253 cases after parametrisation), none of which require network access:

- **CI**: `.github/workflows/ci.yml` runs the full test suite on every push/PR (Python 3.11 + 3.12), then compile-checks all source modules. Status: [![CI](https://github.com/ZXTLQQ/FactorGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/ZXTLQQ/FactorGPT/actions/workflows/ci.yml)
- **Core tests**: sandbox security & lookahead-bias rejection (`test_sandbox.py`, 15 test functions / 24 cases including parametrized future-column names), the six-stage refinery pipeline end-to-end (`test_refinery.py`, 6), and documentation-contract drift guards (`test_docs_contract.py`, 6 — keeps README page/factor counts, the `instrument→symbol` data contract, and `kronos.fallback_to_stub` from silently drifting).
- **Backtest math & engineering glue** (`test_backtest.py`, 8 / `test_engineering.py`, 5): IC and rank-IC sign conventions, turnover consistency, lookahead detection on unshifted or negatively-shifted prices, an AlphaLens cross-check, plus experiment tracking, GP mining, LLM routing, batch evaluation and HPO.
- **Forward-testing bridge** (`test_forwardtest.py`, 29): the deterministic macro-theme → asset/direction/confidence translation, append-only ledger semantics with prediction fields frozen once settled, scorecard math (directional accuracy, Brier against the 1/3 random baseline, confidence-bucket calibration), and the guarantee that a missing network, missing credentials, or a theme without macro wording degrades to a local dry-run record rather than an exception.
- **Ablation experiments**: `python scripts/ablation_study.py --seed 42 --n-symbols 20` quantifies each pipeline module's marginal contribution on out-of-sample data (ΔICIR per module); results and interpretation in [docs/ablation_report.md](docs/ablation_report.md).
- **Factor research layer**: `tests/test_mining.py` (32 tests) pins the operator library against pandas / hand-computed references, the type-gate rejections, PIT alignment vs the independent reference implementation, warm-up-aware coverage, the risk-gate identities, grid-miner budget & reproducibility, the expected direction of every fundamental / concept factor, and the report renderer (every section present, `—` instead of `nan`, escaped table pipes, byte-identical output on write).
- **AI advisor & system diagnostics** (`test_system_advisor.py`, 24 functions / 36 cases): builds a real factor system on a synthetic panel with spectral cleaning switched on, then pins the advisor contract — no `nan` can enter the fact table, every listed intent produces an answer, suggested questions must route to a *specific* intent (a suggestion that falls through to the generic one wastes a turn), actions stay ordered and capped, the prompt carries the fact table and forbids numbers outside it, and a model that raises **or** returns blank degrades to the rule answer with the reason attached. The same file covers the concentrated-risk branch of `factor_system.build_findings`, whose input is the risk decomposition above. Uncovered surface is stated rather than implied: three of the six modules in Highlight 10 (significance / universe / multi-scale GP) are covered by `tests/test_triage.py`, `specification_rl` has its own file (below), `param_ops` runs only inside the multi-scale miner (exercised by implication), and the spectral identities (`mp_bounds`, shrinkage) still have no dedicated numeric test — the chain is exercised end-to-end through this fixture instead.
- **Multitask specification RL** (`test_specification_rl.py`, 14 tests): the three action masks, that a compiled specification is a valid GP expression tree (compiled with the same tuple grammar and evaluated by `eval_tree`), that the panel environment scores `|IC|` and fails a specification whose columns are missing, that the reward is bounded and returns `−1` on an unestimable specification, and — the falsifiable one — that the shared policy beats single-task training **and** an untrained policy on held-out tasks whose features were never seen in training (0.75 / 0.19 / 0.32, 8/8 seeds). Coverage is also pinned end-to-end: the report renders the specification section, and `multiscale_mine(spec_rl=True)` actually injects the RL seeds into the genetic initial population rather than carrying them alongside.
- **LLM provenance & config persistence** (`test_llm_provenance.py`, 11 / `test_config_persistence.py`, 22 tests): the two failure modes that made "the model is connected but it keeps running offline" unexplainable. An unresolved `${VAR}` key must be rejected **at call time** (construction must survive it, or the whole Streamlit app dies on a missing env var) and `_build()`'s refusal must be visible in `available()`; the offline fallback must record `factor_source` / `llm_error` in state, merge the reflect-stage failure with the generate-stage one instead of overwriting it, and stamp the report header so a template product cannot pass as model output. On the persistence side: a plaintext key must never reach the git-tracked `config.yaml` (it goes to `.env`, and the placeholder must still interpolate back to the real value), saving must be a line-level patch that touches only the target section — including the same-named key one level deeper in `llm.router.critic` and the top-level `proxy` vs `data.proxy` pair — and a save that changes nothing must leave the file byte-identical.
- **Warnings are errors** (`pytest.ini`): the failure mode this repo cares about is not a raised exception but an *exception silently swallowed into a plausible number* — `np.corrcoef` returning 0 for a degenerate cross-section, or `np.nanmean` warning on an empty slice and quietly yielding NaN. Every `RuntimeWarning` / `FutureWarning` / `DeprecationWarning` now fails the suite, so those substitutions cannot creep back in.

---

## Roadmap

- [x] Forward-testing channel for factor-layer macro views (Headline Arena; locked-before-outcome predictions, third-party settlement — see Highlight 9)
- [x] Typed factor research layer (expression DSL, PIT-safe panel, operator grid miner, four-dimension evaluation — see Highlight 10)
- [x] Spectral cleaning and risk decomposition of the factor correlation matrix (see Highlight 12)
- [x] Wire the significance gates, the universe labels and the multi-scale GP into the mining flow, the report and the UI, and cover them with tests (see Highlight 10)
- [x] Multitask RL specification search — shared DeepSet Q-policy, zero-shot transfer to unseen features, seeding the GP initial population (see Highlight 13)
- [ ] Surface the parameterised temporal-memory operator (`hwma`, Highlight 10) as a first-class operator in the expression DSL — today it runs only inside the multi-scale miner
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
