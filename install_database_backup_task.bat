@echo off
title Install Pharmacy Pre-closing Backup
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-database-backup-task.ps1"
if errorlevel 1 (
    echo.
    echo Backup-task installation failed. Review the message above.
    pause
    exit /b 1
)
echo.
echo Database backups are now scheduled one hour before closing on open business days.
pause
