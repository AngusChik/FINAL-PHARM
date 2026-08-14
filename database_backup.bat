@echo off
title Pharmacy Database Backup
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\database-backup.ps1" -Reason manual
if errorlevel 1 (
    echo.
    echo Database backup failed. Nothing was deleted or replaced.
    pause
    exit /b 1
)
echo.
echo Backup completed and verified.
pause
