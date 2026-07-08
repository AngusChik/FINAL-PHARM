@echo off
cd /d "%~dp0"
if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
) else if exist "..\env\Scripts\activate.bat" (
    call ..\env\Scripts\activate.bat
)
start http://127.0.0.1:8000
python manage.py runserver
