@echo off
setlocal
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 goto missing_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python

if not exist ".venv\Scripts\python.exe" (
    echo Creating a private Python environment for this app...
    python -m venv .venv
    if errorlevel 1 goto failed
)

echo Checking the app's packages...
".venv\Scripts\python.exe" -m pip install -e ".[app]"
if errorlevel 1 goto failed

echo Opening the local forecasting app. Press Ctrl+C here to stop it.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
exit /b %errorlevel%

:missing_python
echo Python 3.11 or newer is required. Install Python, then run START_APP.bat again.
pause
exit /b 1

:failed
echo Setup did not finish. Read docs\getting-started.md for help.
pause
exit /b 1
