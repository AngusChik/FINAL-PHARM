@echo off
title Pharmacy Server - Control Panel
color 0B

:MENU
cls
echo.
echo  ============================================
echo       PHARMACY APP - CONTROL PANEL
echo  ============================================
echo.
echo  Server Address: http://192.168.0.15:8000
echo.
echo  [1] Start Server
echo  [2] Stop Server
echo  [3] Check Server Status
echo  [4] Open App in Browser
echo  [5] Create a New Staff Account
echo  [6] Open Firewall Port (first-time setup)
echo  [7] Close Firewall Port
echo  [8] Exit
echo.
echo  ============================================
set /p choice="  Enter choice (1-8): "

if "%choice%"=="1" goto START
if "%choice%"=="2" goto STOP
if "%choice%"=="3" goto STATUS
if "%choice%"=="4" goto BROWSER
if "%choice%"=="5" goto NEWUSER
if "%choice%"=="6" goto FIREWALL_OPEN
if "%choice%"=="7" goto FIREWALL_CLOSE
if "%choice%"=="8" goto EXIT
echo  Invalid choice.
pause
goto MENU

:START
cls
echo.
echo  Starting server...
echo.
cd /d "%~dp0"
call env\Scripts\activate.bat
python manage.py collectstatic --noinput >nul 2>&1
echo  Static files collected.
echo.
echo  ============================================
echo   SERVER IS RUNNING
echo   Address: http://192.168.0.15:8000
echo   Press Ctrl+C to stop, then any key to
echo   return to the control panel.
echo  ============================================
echo.
waitress-serve --host=0.0.0.0 --port=8000 --threads=4 inventory.wsgi:application
echo.
echo  Server stopped.
pause
goto MENU

:STOP
cls
echo.
echo  Stopping server...
taskkill /f /im waitress-serve.exe >nul 2>&1
if %errorlevel%==0 (
    echo  Server stopped successfully.
) else (
    echo  Server was not running.
)
echo.
pause
goto MENU

:STATUS
cls
echo.
echo  Checking server status...
echo.
netstat -an | findstr ":8000" >nul 2>&1
if %errorlevel%==0 (
    echo  [RUNNING] Server is active on port 8000
    echo.
    echo  Active connections:
    netstat -an | findstr ":8000"
) else (
    echo  [STOPPED] Server is not running
)
echo.
pause
goto MENU

:BROWSER
start http://192.168.0.15:8000
goto MENU

:NEWUSER
cls
echo.
echo  Create a New Staff Account
echo  ============================================
echo.
cd /d "%~dp0"
call env\Scripts\activate.bat
python manage.py createsuperuser
echo.
pause
goto MENU

:FIREWALL_OPEN
cls
echo.
echo  Opening firewall port 8000 for LAN...
echo  (This requires Administrator privileges)
echo.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Run this control panel as Administrator
    echo  Right-click server_control.bat ^> Run as administrator
    echo.
    pause
    goto MENU
)
netsh advfirewall firewall add rule name="Pharmacy App" dir=in action=allow protocol=TCP localport=8000 remoteip=192.168.0.0/24
echo.
echo  Firewall rule added. Other computers can now connect.
echo.
pause
goto MENU

:FIREWALL_CLOSE
cls
echo.
echo  Closing firewall port 8000...
echo.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Run this control panel as Administrator
    echo  Right-click server_control.bat ^> Run as administrator
    echo.
    pause
    goto MENU
)
netsh advfirewall firewall delete rule name="Pharmacy App"
echo.
echo  Firewall rule removed. Other computers can no longer connect.
echo.
pause
goto MENU

:EXIT
exit
