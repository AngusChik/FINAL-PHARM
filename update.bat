@echo off
title Pharmacy App - Update and Restart
color 0E
REM One-click update: stop server, apply DB migrations, refresh static files, restart.
REM Run this after pulling/copying new code so the running server picks up changes.

cd /d "%~dp0"
call env\Scripts\activate.bat

echo.
echo  ============================================
echo    PHARMACY APP - UPDATE
echo  ============================================
echo.

echo  [1/4] Stopping server (if running)...
taskkill /f /im waitress-serve.exe >nul 2>&1
if %errorlevel%==0 (echo        stopped.) else (echo        not running.)
echo.

echo  [2/4] Applying database migrations...
python manage.py migrate --noinput
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: migrations failed. Fix the error above, then re-run update.bat.
    echo  The server was NOT restarted.
    pause
    exit /b 1
)
echo.

echo  [3/4] Collecting static files...
python manage.py collectstatic --noinput >nul 2>&1
echo        done.
echo.

echo  [4/4] Starting server...
echo  ============================================
echo   SERVER IS RUNNING
echo   Address: http://192.168.0.15:8000
echo   Press Ctrl+C to stop.
echo  ============================================
echo.
waitress-serve --host=0.0.0.0 --port=8000 --threads=4 inventory.wsgi:application
echo.
echo  Server stopped.
pause
