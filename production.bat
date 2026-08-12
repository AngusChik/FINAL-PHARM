@echo off
title Pharmacy Production Control
cd /d "%~dp0"
if "%~1"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\production.ps1" -Action menu
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\production.ps1" %*
)
if errorlevel 1 (
    echo.
    echo Production command failed. Review the message above and the logs folder.
    pause
)
