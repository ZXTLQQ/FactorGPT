# ============================================================
# FactorGPT Startup Script (ASCII safe)
# ============================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FactorGPT - LLM Factor Mining Agent" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.10+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check dependencies
Write-Host "[*] Checking dependencies..." -ForegroundColor Yellow
$null = python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "[*] Starting Streamlit server..." -ForegroundColor Yellow
Write-Host "    Local:   http://localhost:8501" -ForegroundColor White
Write-Host "    Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 国内 HuggingFace 镜像：避免知识库下载 BGE 向量模型时连接 huggingface.co 超时
# （WinError 10060）。如网络可直连官方源，可注释下面两行或将值改为 https://huggingface.co
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "60"

python -m streamlit run src/ui/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true

Read-Host "Server stopped. Press Enter to close"
