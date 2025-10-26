@echo off
cd /d %~dp0

REM Aktivera virtuell miljö (venv312) om den finns
if exist venv312\Scripts\activate (
    call venv312\Scripts\activate
) else (
    echo ❌ Ingen virtuell miljö hittades. Kör install.bat först!
    pause
    exit /b
)

:MENU
cls
echo ============================
echo   STRYK – STARTMENY
echo ============================
echo [1] Hämta ligadata
echo [2] Träna modeller
echo [3] Starta app
echo [Q] Avsluta
echo.
set /p choice=Välj ett alternativ: 

if "%choice%"=="1" goto FETCH
if "%choice%"=="2" goto TRAIN
if "%choice%"=="3" goto APP
if /I "%choice%"=="Q" goto END
goto MENU

:FETCH
echo ============================
echo Hämtar ligadata...
echo ============================
python src/fetch_ligor.py
pause
goto MENU

:TRAIN
echo ============================
echo Tränar modeller...
echo ============================
python src/train.py
pause
goto MENU

:APP
echo ============================
echo Startar Streamlit-appen...
echo ============================
streamlit run src/app.py
goto END

:END
echo.
echo Hejdå!
pause
