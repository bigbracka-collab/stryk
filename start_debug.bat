@echo off
chcp 65001 >nul
title Fotboll-AI (Smart Start)

:: ===========================
:: FOTBOLL-AI – STARTMENY
:: Python 3.12 + venv312
:: ===========================

cd /d "%~dp0"

:: --- Kontrollera src/app.py finns ---
if not exist "src\app.py" (
  echo.
  echo [ERROR] src\app.py hittas inte. Är du i projektroten?
  echo.
  pause
  exit /b 1
)

:: --- Kontrollera virtuell miljö ---
if not exist "venv312\Scripts\activate.bat" (
  echo.
  echo [ERROR] Virtuell miljö saknas!
  echo Kör: reset_venv.cmd eller install.bat för att skapa den.
  echo.
  pause
  exit /b 1
)

:: --- Aktivera miljön ---
call "venv312\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Kunde inte aktivera venv312.
  pause
  exit /b 1
)

:: --- Kontrollera Python och Pip ---
echo [INFO] Python-version:
python -c "import platform; print(platform.python_version())"
if errorlevel 1 (
  echo [ERROR] Python saknas i venv.
  pause
  exit /b 1
)

:: --- Kolla så viktiga paket finns ---
echo [INFO] Kontrollerar Streamlit & Dotenv...
python -c "import streamlit, dotenv" >nul 2>&1
if errorlevel 1 (
  echo [WARN] Saknar paket – installerar nödvändiga bibliotek...
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install streamlit==1.39.0 python-dotenv==1.0.1 pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 joblib==1.4.2 matplotlib==3.9.2 seaborn==0.13.2 scipy==1.14.1
)

:: --- Bekräfta att import nu fungerar ---
python -c "import streamlit, dotenv; print('✅ Alla paket OK!')" || (
  echo [ERROR] Paketinstallation misslyckades.
  pause
  exit /b 1
)

:: --- Meny ---
:MENU
cls
echo.
echo [INFO] FOTBOLL-AI – HUVUDMENY
echo ==========================================
echo   [1] Hämta ligadata
echo   [2] Träna modeller
echo   [3] Starta Streamlit-appen
echo   [4] Uppdatera paket
echo   [Q] Avsluta
echo ==========================================
echo.
set /p choice=Välj [1-4 eller Q]: 

if /I "%choice%"=="1" goto FETCH
if /I "%choice%"=="2" goto TRAIN
if /I "%choice%"=="3" goto APP
if /I "%choice%"=="4" goto INSTALL
if /I "%choice%"=="Q" goto END
goto MENU

:FETCH
cls
echo [INFO] Hämtar ligadata...
python -m src.fetch_ligor
pause
goto MENU

:TRAIN
cls
echo [INFO] Tränar modeller...
python -m src.train
pause
goto MENU

:APP
cls
echo [INFO] Startar Streamlit-app...
echo URL: http://localhost:8501
echo.

REM -- Öppna Chrome automatiskt --
start "" "chrome" "http://localhost:8501"

REM -- Kör Streamlit via python -m (garanterar rätt venv) --
python -m streamlit run src\app.py --server.port=8501
goto END

:INSTALL
cls
echo [INFO] Uppdaterar alla paket...
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
pause
goto MENU

:END
echo.
echo [SUCCESS] Hejdå! Tack för att du använder Fotboll-AI!
echo.
pause
exit /b 0
