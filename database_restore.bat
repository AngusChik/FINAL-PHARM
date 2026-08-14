@echo off
title Pharmacy Database Restore
cd /d "%~dp0"
if "%~1"=="" (
    echo Usage: database_restore.bat "C:\path\to\pharmacy-backup.dump"
    echo.
    echo Stop production first with: production.bat stop
    pause
    exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\database-restore.ps1" -BackupPath "%~1"
if errorlevel 1 (
    echo.
    echo Restore did not complete. Review the message above.
    pause
    exit /b 1
)
pause
