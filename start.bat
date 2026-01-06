@echo off
cd C:\Users\Angus\Desktop\Pharm-main
call env\Scripts\Activate
start http://127.0.0.1:8000
python manage.py runserver