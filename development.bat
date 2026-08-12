@echo off
title Pharmacy Development Control
cd /d "%~dp0"
if "%~1"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\development.ps1" -Action menu
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\development.ps1" %*
)
if errorlevel 1 (
    echo.
    echo Development startup failed.
    pause
)
