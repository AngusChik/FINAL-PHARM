@echo off
REM ============================================================
REM  Daily end-of-day pharmacy report.
REM  Point a Windows Task Scheduler task at this file (e.g. run
REM  it once daily around closing time, ~6:00 PM). It builds the
REM  digest + PDF and, once DAILY_REPORT_RECIPIENTS + EMAIL_* are
REM  configured, emails it. Until then it just logs the digest.
REM ============================================================
cd /d "%~dp0"
if not exist logs mkdir logs
call env\Scripts\activate.bat
python manage.py send_daily_report >> logs\daily_report.log 2>&1
