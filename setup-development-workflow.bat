@echo off
setlocal
title Pharmacy Development-First Setup
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-development-workflow.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"

echo.
if not "%SETUP_EXIT%"=="0" (
    echo Development-first setup failed. Read the error above.
) else (
    echo Development-first setup completed successfully.
)
echo.
pause
exit /b %SETUP_EXIT%
