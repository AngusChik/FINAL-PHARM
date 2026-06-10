@echo off
title Pharmacy Server - RUNNING
color 0A

echo.
echo  ============================================
echo       PHARMACY APP - LOCAL NETWORK SERVER
echo  ============================================
echo.
echo  Status:  STARTING...
echo  Address: http://192.168.0.15:8000
echo.
echo  Other computers on the network open:
echo     http://192.168.0.15:8000
echo.
echo  ============================================
echo   To STOP the server: press Ctrl+C or close
echo   this window
echo  ============================================
echo.

cd /d "%~dp0"
call env\Scripts\activate.bat

echo  Collecting static files...
python manage.py collectstatic --noinput >nul 2>&1
echo  Done.
echo.
echo  Server is RUNNING. Do not close this window.
echo  ============================================
echo.

waitress-serve --host=0.0.0.0 --port=8000 --threads=4 inventory.wsgi:application
