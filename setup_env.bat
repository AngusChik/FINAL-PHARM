@echo off
echo Installing virtualenv...
pip install virtualenv

echo Creating virtual environment...
python -m virtualenv env

echo Activating virtual environment...
call env\Scripts\activate.bat

echo Installing packages...
pip install django
pip install psycopg2-binary
pip install python-dateutil
pip install reportlab

echo ✅ Environment setup complete!
pause
