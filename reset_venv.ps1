# reset_venv.ps1
# Skapar om venv312 (Python 3.12), installerar exakta paket och verifierar installation.

$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Ok($m)  { Write-Host "[OK]    $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Err($m) { Write-Host "[ERROR] $m" -ForegroundColor Red }

# 0) Kör från scriptets mapp (robust)
$scriptRoot = $PSScriptRoot
if (-not $scriptRoot) {
    $scriptRoot = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
}
Set-Location -Path $scriptRoot

# 1) Ta bort gammal venv
$venvPath = Join-Path (Get-Location) "venv312"
if (Test-Path $venvPath) {
    Info "Tar bort gammal venv: $venvPath"
    Remove-Item -Recurse -Force $venvPath
    Ok "Gammal venv borttagen."
} else {
    Info "Ingen gammal venv hittades. Fortsätter."
}

# 2) Skapa ny venv (py -3.12 först, sedan fallback)
Info "Skapar ny venv (försöker 'py -3.12')..."
py -3.12 -m venv venv312 2>$null
if (-not (Test-Path "venv312\Scripts\python.exe")) {
    Warn "'py -3.12' misslyckades eller saknas. Försöker 'python -m venv'..."
    python -m venv venv312
}
if (-not (Test-Path "venv312\Scripts\python.exe")) {
    Err "Kunde inte skapa venv312. Är Python 3.12 installerad?"
    exit 1
}
Ok "venv312 skapad."

# 3) Aktivera venv (bypass i denna process)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null
. ".\venv312\Scripts\Activate.ps1"

# 4) Uppgradera pip/setuptools/wheel
Info "Uppgraderar pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# 5) Stäng av pip-versionvarning (valfritt men skönt)
Info "Stänger av pip version check..."
pip config set global.disable-pip-version-check true

# 6) Installera exakta versioner (pinned)
Info "Installerar exakta paketversioner..."
$pkgs = @(
  "streamlit==1.39.0",
  "pandas==2.2.3",
  "numpy==1.26.4",
  "scikit-learn==1.5.2",
  "joblib==1.4.2",
  "matplotlib==3.9.2",
  "seaborn==0.13.2",
  "scipy==1.14.1",
  "python-dotenv==1.0.1"
)
pip install $pkgs

# 7) Verifiering via temporär .py
Info "Verifierar installation..."
$pycode = @'
import sys, platform
print("Python:", sys.executable)
print("Version:", platform.python_version())
import streamlit, pandas, numpy, sklearn, joblib, matplotlib, seaborn, scipy, dotenv  # noqa: F401
print("Imports OK.")
'@
$tmpPy = Join-Path $env:TEMP "venv_verify.py"
Set-Content -Path $tmpPy -Value $pycode -Encoding ASCII
python $tmpPy
$code = $LASTEXITCODE
Remove-Item -Force $tmpPy -ErrorAction SilentlyContinue

if ($code -ne 0) {
    Err "Verifiering misslyckades (något paket saknas?)."
    exit 1
}

Ok "Klar! venv312 är redo."
Write-Host "Tips:"
Write-Host "  .\venv312\Scripts\activate"
Write-Host "  streamlit run src\app.py --server.port=8501"
