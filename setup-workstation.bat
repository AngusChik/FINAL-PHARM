@echo off
setlocal
title Pharmacy Workstation Setup
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-workstation.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"

echo.
if not "%SETUP_EXIT%"=="0" (
    echo Workstation setup failed. Read the error above.
) else (
    echo Workstation setup completed successfully.
)
echo.
pause
exit /b %SETUP_EXIT%
