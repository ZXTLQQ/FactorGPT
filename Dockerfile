# FactorGPT 运行镜像（评委机器一键复现）
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# 系统依赖（sentence-transformers / chromadb 等可能需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用层缓存）
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# 复制源码与配置
COPY . /app

# 数据/缓存目录
RUN mkdir -p /app/data/cache /app/chroma_db

EXPOSE 8501

# 默认启动 Streamlit（单因子 Agent / 精炼厂 / 监控看板等）
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
