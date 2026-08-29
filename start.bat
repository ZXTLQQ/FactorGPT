@echo off
rem ============================================================
rem  FactorGPT startup batch (double-click to run)
rem  All logic lives in start.ps1; this file only calls it.
rem  Edit start.ps1 to change startup parameters.
rem ============================================================

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] FactorGPT failed to start (exit code %EXIT_CODE%).
    echo         See the log above for details.
    echo.
)

pause
