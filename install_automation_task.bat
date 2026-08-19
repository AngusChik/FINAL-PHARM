@echo off
title Install Pharmacy Scheduled Jobs
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-automation-task.ps1"
if errorlevel 1 (
    echo.
    echo Automation-task installation failed. Review the message above.
    pause
    exit /b 1
)
echo.
echo Pharmacy automation is installed and checks for due work hourly in the background.
pause
