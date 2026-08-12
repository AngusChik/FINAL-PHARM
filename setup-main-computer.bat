@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Pharmacy Main Computer - One-Time Setup

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-main-computer.ps1"
set "SETUP_EXIT=%ERRORLEVEL%"

echo.
if not "%SETUP_EXIT%"=="0" (
    echo One-time setup did not complete. Read the error above.
) else (
    echo One-time setup completed successfully. Starting production...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\production.ps1" -Action start
    set "SETUP_EXIT=!ERRORLEVEL!"
    if not "!SETUP_EXIT!"=="0" echo Production startup failed. Read the error above.
)
echo.
pause
exit /b !SETUP_EXIT!
