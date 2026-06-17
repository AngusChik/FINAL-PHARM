@echo off
REM Run the Django test suite using the project's virtualenv.
REM Tests run against a throwaway "test_<db>" database — your real data is never touched.
REM
REM   run_tests.bat              -> run all app tests
REM   run_tests.bat CheckoutTests-> run a single test class
REM   run_tests.bat CheckoutTests.test_submit_decrements_stock_once_and_records_change  -> one test

cd /d "%~dp0"

if "%~1"=="" (
    env\Scripts\python.exe manage.py test app --noinput
) else (
    env\Scripts\python.exe manage.py test app.tests.%~1 --noinput
)

echo.
pause
