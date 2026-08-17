@echo off
title Install Pharmacy Database Backup Task
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-database-backup-task.ps1"
if errorlevel 1 (
    echo.
    echo Backup-task installation failed. Review the message above.
    pause
    exit /b 1
)
echo.
echo Daily backups are now scheduled for 2:00 AM.
pause
