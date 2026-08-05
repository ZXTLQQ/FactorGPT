# FactorGPT — LLM-Powered Quantitative Factor Mining & Industrialization Platform
# 
# Build:  docker build -t factorgpt:latest .
# Run:    docker-compose up -d
# Quick:  docker run -d -p 8501:8501 factorgpt:latest
#
# Environment:
#   DEEPSEEK_API_KEY   - DeepSeek API key (optional, defaults to offline/synthetic)
#   OPENAI_API_KEY     - OpenAI API key (optional)
#   TUSHARE_TOKEN      - Tushare Pro token (optional, for real A-share data)
#   FACTORGPT_LLM_PROVIDER - Override LLM provider (default: ollama)
#   FACTORGPT_LLM_MODEL    - Override LLM model (default: qwen2.5-coder:7b)
#   HF_ENDPOINT        - HuggingFace mirror (default: https://hf-mirror.com)

FROM python:3.11-slim

# ============================================================
# Environment Configuration
# ============================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# ============================================================
# System Dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python Dependencies (Layer Cache Optimization)
# ============================================================
# Prefer locked requirements for reproducible builds
COPY requirements.lock.txt /app/requirements.lock.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.lock.txt 2>/dev/null \
    || pip install -r requirements.txt

# ============================================================
# Application Source
# ============================================================
COPY . /app

# Create data and cache directories
RUN mkdir -p /app/data/cache /app/chroma_db

# ============================================================
# Health Check
# ============================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ============================================================
# Expose & Run
# ============================================================
EXPOSE 8501

# Default: Streamlit 17-page web interface
# Override CMD to run CLI: docker run factorgpt python run_agent.py "..."
CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
