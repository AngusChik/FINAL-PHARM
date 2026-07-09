@echo off
REM ============================================================
REM  Fill the McKesson PharmaClik cart from Recently Purchased
REM  products. Opens a browser; you review and submit the order.
REM  Extra arguments are passed through, e.g.:
REM      order_mckesson.bat --dry-run
REM      order_mckesson.bat --limit 2
REM ============================================================
cd /d "%~dp0"
call env\Scripts\activate.bat
python mckesson_order.py %*
pause
