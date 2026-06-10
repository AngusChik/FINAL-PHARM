@echo off
title Pharmacy Server - HTTPS (waitress + Caddy)
color 0E

echo.
echo  ============================================
echo     PHARMACY APP - SECURE (HTTPS) SERVER
echo  ============================================
echo.
echo  Address: https://192.168.0.15
echo.
echo  This starts waitress bound to localhost only, then
echo  starts Caddy which terminates TLS on port 443.
echo  See DEPLOYMENT_HTTPS.md for first-time setup.
echo  ============================================
echo.

cd /d "%~dp0"
call env\Scripts\activate.bat

REM Enable Django's secure cookie / SSL settings for this process tree
set DJANGO_SECURE=1

echo  Collecting static files...
python manage.py collectstatic --noinput >nul 2>&1
echo  Done.
echo.

REM waitress listens on localhost only -> reachable ONLY through Caddy
echo  Starting waitress on 127.0.0.1:8000 (internal)...
start "Pharmacy waitress (internal)" /min cmd /c "set DJANGO_SECURE=1 && waitress-serve --host=127.0.0.1 --port=8000 --threads=4 inventory.wsgi:application"

REM Give waitress a moment to bind
ping -n 3 127.0.0.1 >nul

echo  Starting Caddy (HTTPS :443)...
echo  ============================================
echo   SECURE SERVER RUNNING - https://192.168.0.15
echo   Press Ctrl+C to stop Caddy, then close the
echo   minimized waitress window to fully stop.
echo  ============================================
echo.
caddy run --config Caddyfile

echo.
echo  Caddy stopped. Remember to also stop the waitress window.
pause
