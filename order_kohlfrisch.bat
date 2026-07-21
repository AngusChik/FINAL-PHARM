@echo off
REM ============================================================
REM  Fill the Kohl & Frisch (KFConnect) cart from Recently
REM  Purchased products. Opens a browser; you review and submit
REM  the order. Extra arguments are passed through, e.g.:
REM      order_kohlfrisch.bat --dry-run
REM      order_kohlfrisch.bat --limit 2
REM ============================================================
cd /d "%~dp0"
call env\Scripts\activate.bat
python kohlfrisch_order.py %*
pause
