@echo off
setlocal
title Pharmacy Startup and Shortcuts Setup
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-production-startup.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"

echo.
if not "%SETUP_EXIT%"=="0" (
    echo Pharmacy startup setup failed. Read the error above.
) else (
    echo Pharmacy startup setup completed successfully.
)
echo.
pause
exit /b %SETUP_EXIT%
