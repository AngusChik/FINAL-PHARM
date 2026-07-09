@echo off
echo Installing virtualenv...
pip install virtualenv

echo Creating virtual environment...
python -m virtualenv env

echo Activating virtual environment...
call env\Scripts\activate.bat

echo Installing packages from requirements.txt...
pip install -r requirements.txt

echo.
echo Installing Playwright + Chromium (for the McKesson ordering tool)...
pip install playwright
python -m playwright install chromium

echo.
echo Configuring this computer's IP address...
call configure_ip.bat

echo Environment setup complete!
pause
