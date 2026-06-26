@echo off
echo ============================================
echo    Configure server IP address
echo ============================================
echo.
echo  Enter the LAN IP address of THIS computer.
echo  Tip: run 'ipconfig' and look for "IPv4 Address".
echo.
set /p NEWIP="  New IP address (e.g. 192.168.1.42): "

if "%NEWIP%"=="" (
  echo.
  echo  No IP entered - nothing changed.
  pause
  exit /b 1
)

REM Prefer the project's venv python if it exists, else fall back to PATH.
if exist "env\Scripts\python.exe" (
  env\Scripts\python.exe configure_ip.py %NEWIP%
) else (
  python configure_ip.py %NEWIP%
)

echo.
pause
