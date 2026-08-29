@echo off
chcp 65001 >nul
rem ============================================================
rem  FactorGPT 启动批处理（可双击运行）
rem  实际逻辑在 start.ps1 中，本文件负责调用它。
rem  如需修改启动参数，请编辑 start.ps1。
rem ============================================================

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] FactorGPT 启动失败（退出码 %EXIT_CODE%）。
    echo         请查看上方日志确认原因。
    echo.
)

pause
