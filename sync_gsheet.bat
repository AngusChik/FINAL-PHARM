@echo off
REM ============================================================
REM  Google Sheet -> Ordering Sheet pull (MANUAL / optional CLI helper).
REM  There is no automatic schedule — staff pull on demand with the
REM  "Pull from Google Sheet" button on the Ordering Sheet page. This .bat
REM  just runs the same pull from a terminal if ever needed.
REM  Does nothing until GSHEET_SPREADSHEET_ID is set in .env and
REM  google_credentials.json exists.
REM ============================================================
cd /d "%~dp0"
if not exist logs mkdir logs
call env\Scripts\activate.bat
python manage.py sync_gsheet >> logs\gsheet_sync.log 2>&1
