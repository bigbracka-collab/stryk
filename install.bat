@echo off
cd /d %~dp0

echo ============================
echo   STRYK – INSTALLATION
echo ============================

REM Skapa virtuell miljö med Python 3.12
if not exist venv312 (
    echo 🔧 Skapar virtuell miljö med Python 3.12...
    py -3.12 -m venv venv312
) else (
    echo ✅ Virtuell miljö (venv312) finns redan.
)

REM Aktivera miljön
call venv312\Scripts\activate

REM Uppdatera pip och installera paket
echo 📦 Installerar paket från requirements.txt...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo.
echo ✅ Klar! Miljön (venv312) är redo.
pause
