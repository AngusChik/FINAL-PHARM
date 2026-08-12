@echo off
setlocal
cd /d "%~dp0"

echo [1/5] Creating the virtual environment...
if not exist "env\Scripts\python.exe" python -m venv env
if errorlevel 1 goto :FAIL

echo [2/5] Updating pip...
env\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :FAIL

echo [3/5] Installing application dependencies...
env\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :FAIL

echo [4/5] Installing Chromium for supplier ordering...
env\Scripts\python.exe -m playwright install chromium
if errorlevel 1 goto :FAIL

echo [5/5] Configuring the server LAN address...
call configure_ip.bat
if errorlevel 1 goto :FAIL

echo.
echo Setup complete. Use development.bat or production.bat.
pause
exit /b 0

:FAIL
echo.
echo Setup failed. Fix the error above and run setup_env.bat again.
pause
exit /b 1
