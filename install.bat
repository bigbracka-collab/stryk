@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: =============================================
:: FOTBOLL-AI – INSTALLATION (Windows, Python 3.12)
:: =============================================

chcp 65001 >nul
title Fotboll-AI – Installation

:: Gå till projektroten (där denna fil ligger)
cd /d "%~dp0"

cls
echo.
echo =============================================
echo   FOTBOLL-AI – INSTALLATION
echo =============================================
echo.

:: [1] Hitta Python 3.12 (py launcher först, fallback till python)
set "PY_CMD="
py -3.12 --version >nul 2>&1 && set "PY_CMD=py -3.12"
if "%PY_CMD%"=="" (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
        rem %%v blir t.ex. 3.12.6
        set "VER=%%v"
    )
    echo %VER% | findstr /b "3.12" >nul && set "PY_CMD=python"
)
if "%PY_CMD%"=="" (
    echo [ERROR] Python 3.12 hittades inte.
    echo         Installera Python 3.12 och prova igen:
    echo         https://www.python.org/downloads/release/python-3120/
    echo.
    pause
    exit /b 1
)
echo [OK] Python-kommando: %PY_CMD%

:: [2] Skapa venv om saknas
if not exist "venv312\Scripts\activate.bat" (
    echo.
    echo Skapar virtuell miljo (venv312)...
    %PY_CMD% -m venv venv312
    if errorlevel 1 (
        echo [ERROR] Kunde inte skapa venv312
        pause
        exit /b 1
    )
) else (
    echo [OK] venv312 finns redan
)

:: [3] Aktivera venv
call "venv312\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Kunde inte aktivera venv312
    pause
    exit /b 1
)
echo [OK] venv aktiverad

:: [4] Uppgradera pip/setuptools/wheel
echo.
echo Uppgraderar pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel >nul
if errorlevel 1 (
    echo [WARN] Misslyckades uppgradera pip/setuptools/wheel (fortsaetter).
) else (
    echo [OK] pip/setuptools/wheel uppgraderade
)

:: [5] Kontrollera requirements.txt
if not exist "requirements.txt" (
    echo.
    echo [ERROR] requirements.txt saknas i projektroten.
    echo         Skapa filen och koer om installationen.
    echo.
    pause
    exit /b 1
)

:: [6] Installera paket
echo.
echo Installerar paket fran requirements.txt ...
pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo.
    echo [ERROR] Installation misslyckades.
    echo        Prova manuellt:
    echo        venv312\Scripts\activate
    echo        pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo [OK] Paket installerade

:: [7] Skapa projektmappar
echo.
for %%D in (data modeller param logs src) do (
    if not exist "%%D" mkdir "%%D" >nul 2>&1
)
if not exist "src\__init__.py" (
    echo.> "src\__init__.py"
)
echo [OK] Mappar klara: data, modeller, param, logs, src

:: [8] Skapa .env om saknas (stub med ODDS_API_KEY)
if not exist ".env" (
    (
        echo # Miljovariabler for Fotboll-AI
        echo # Fyll i din The Odds API-nyckel nedan:
        echo ODDS_API_KEY=
    ) > ".env"
    echo [OK] Skapade .env (glom inte att fylla i ODDS_API_KEY)
) else (
    echo [OK] .env finns redan
)

:: [9] Verifiera kritiska paket
echo.
echo Verifierar installation...
python - <<PYCODE
import sys
mods = ["streamlit","pandas","numpy","scikit_learn","joblib","matplotlib","seaborn","scipy"]
missing = []
for m in mods:
    try:
        __import__(m if m!="scikit_learn" else "sklearn")
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("MISSING:", missing); sys.exit(1)
print("ALL_OK")
PYCODE
if errorlevel 1 (
    echo [WARN] Nagra paket kunde inte verifieras. Se ovan.
) else (
    echo [OK] Alla paket kunde importeras
)

:: [10] Slut
echo.
echo =============================================
echo   INSTALLATION KLAR!
echo   Koer nu: start.bat
echo   App: http://localhost:8501
echo =============================================
echo.
pause
endlocal
exit /b 0
