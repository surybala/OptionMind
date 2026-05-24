@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo   OptionWheel ^| Setup ^& Launch
echo ============================================
echo.

:: ── 1. Locate Python 3.10+ ──────────────────────────────────────────────────
set PYTHON=

for %%C in (python py python3) do (
    if "!PYTHON!"=="" (
        where %%C >nul 2>&1
        if not errorlevel 1 (
            %%C -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
            if not errorlevel 1 set PYTHON=%%C
        )
    )
)

if "!PYTHON!"=="" (
    for %%P in (
        "%LocalAppData%\Python\bin\python.exe"
        "%LocalAppData%\Programs\Python\Python313\python.exe"
        "%LocalAppData%\Programs\Python\Python312\python.exe"
        "%LocalAppData%\Programs\Python\Python311\python.exe"
        "%LocalAppData%\Programs\Python\Python310\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
    ) do (
        if "!PYTHON!"=="" (
            if exist %%P (
                %%P -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
                if not errorlevel 1 set PYTHON=%%P
            )
        )
    )
)

if "!PYTHON!"=="" (
    echo [ERROR] Python 3.10 or newer not found.
    echo         Download from https://python.org and re-run this script.
    pause
    exit /b 1
)

echo [OK]  Python: !PYTHON!
!PYTHON! --version
echo.

:: ── 2. Create virtual environment ────────────────────────────────────────────
if not exist ".venv" (
    echo [INFO] Creating virtual environment at .venv ...
    !PYTHON! -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]  Virtual environment created.
) else (
    echo [OK]  Virtual environment already exists.
)
echo.

set VENV_PYTHON=.venv\Scripts\python.exe
set VENV_PIP=.venv\Scripts\pip.exe

:: ── 3. Install / upgrade dependencies ────────────────────────────────────────
echo [INFO] Upgrading pip ...
!VENV_PYTHON! -m pip install --upgrade pip --quiet

echo [INFO] Installing dependencies from requirements.txt ...
!VENV_PIP! install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check the output above.
    pause
    exit /b 1
)
echo [OK]  All dependencies installed.
echo.

:: ── 4. Ensure data directory exists ──────────────────────────────────────────
if not exist "data" mkdir data
echo [OK]  data/ directory ready.
echo.

:: ── 5. Validate Alpaca credentials in config.json ────────────────────────────
echo [INFO] Checking Alpaca credentials in config.json ...
!VENV_PYTHON! -c ^"
import json, sys
try:
    cfg = json.load(open('config.json'))
except Exception as e:
    print(f'[ERROR] Cannot read config.json: {e}')
    sys.exit(1)
a = cfg.get('alpaca', {})
key    = a.get('api_key',    '')
secret = a.get('api_secret', '')
paper  = a.get('paper', True)
missing = [name for name, val in [('api_key', key), ('api_secret', secret)] if not val]
if missing:
    print(f'[WARN]  Missing Alpaca credentials: {missing}')
    print('        Edit config.json and add your api_key and api_secret.')
    print('        The agent will still run but no live orders will be placed.')
else:
    mode = 'PAPER' if paper else 'LIVE ***'
    print(f'[OK]    Credentials present ^| mode: {mode}')
^"
echo.

:: ── 6. Run the test suite ─────────────────────────────────────────────────────
echo [INFO] Running test suite to verify the installation ...
!VENV_PYTHON! -m pytest tests/ -q --tb=short
if errorlevel 1 (
    echo.
    echo [ERROR] One or more tests failed. Resolve the issues above before running
    echo         the agent in live mode.
    pause
    exit /b 1
)
echo.

:: ── 7. Launch OptionWheel ────────────────────────────────────────────────────
echo ============================================
echo   Launching OptionWheel Agent
echo ============================================
echo.
!VENV_PYTHON! agent.py
if errorlevel 1 (
    echo.
    echo [ERROR] agent.py exited with an error ^(see above^).
    pause
    exit /b 1
)

echo.
echo [DONE] OptionWheel finished.
pause
