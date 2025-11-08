# start.ps1 - Football-AI (PowerShell clean, approved verbs)
# ASCII-only, no try/catch, no special chars.

$ErrorActionPreference = 'Stop'

function Info($m){ Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Ok($m)  { Write-Host "[OK]    $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Err($m) { Write-Host "[ERROR] $m" -ForegroundColor Red }

# 0) go to script folder
$scriptRoot = $PSScriptRoot
if (-not $scriptRoot) { $scriptRoot = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent }
Set-Location -Path $scriptRoot

# 1) basic check
if (-not (Test-Path ".\src\app.py")) {
  Err "src\app.py not found. Are you in project root?"
  Pause
  exit 1
}

# 2) venv check
$venvActivate = ".\venv312\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
  Err "virtual env missing: $venvActivate"
  Write-Host "Run reset_venv.cmd to create it." -ForegroundColor DarkGray
  Pause
  exit 1
}

# 3) activate venv (policy bypass)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null
. $venvActivate

if (-not $env:VIRTUAL_ENV) {
  Err "failed to activate venv (VIRTUAL_ENV not set)."
  Pause
  exit 1
}
Ok "activated venv: $env:VIRTUAL_ENV"

# 4) python version (warn if not 3.12)
$pyver = & python -c 'import platform;print(platform.python_version())'
Info "python version: $pyver"
if (-not $pyver.StartsWith('3.12')) {
  Warn "your python is not 3.12 (project tested on 3.12). continuing..."
}

# ---- Ensurers ----
function Ensure-BasePackages {
  Info 'checking streamlit and python-dotenv...'
  & python -c 'import streamlit, dotenv' | Out-Null 2>$null
  if ($LASTEXITCODE -ne 0) {
    Warn 'missing base packages - installing pinned set...'
    & python -m pip install --upgrade pip setuptools wheel
    & python -m pip install `
      streamlit==1.39.0 `
      python-dotenv==1.0.1 `
      pandas==2.2.3 `
      numpy==1.26.4 `
      scikit-learn==1.5.2 `
      joblib==1.4.2 `
      matplotlib==3.9.2 `
      seaborn==0.13.2 `
      scipy==1.14.1
    & python -c 'import streamlit, dotenv' | Out-Null 2>$null
    if ($LASTEXITCODE -ne 0) {
      Err 'base package install failed. run reset_venv.cmd or install manually.'
      Pause
      exit 1
    }
  }
  Ok 'base packages OK.'
}

function Ensure-MLStack {
  Info 'checking ML stack (joblib, sklearn, numpy, scipy, pandas, matplotlib, seaborn)...'
  $pyCheck = @"
import importlib, sys
mods = ["joblib","sklearn","numpy","scipy","pandas","matplotlib","seaborn"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:", ",".join(missing))
    sys.exit(1)
print("OK")
"@
  $pyCheck | & python -
  if ($LASTEXITCODE -ne 0) {
    Warn 'missing ML packages - installing pinned set...'
    & python -m pip install `
      joblib==1.4.2 `
      scikit-learn==1.5.2 `
      numpy==1.26.4 `
      scipy==1.14.1 `
      pandas==2.2.3 `
      matplotlib==3.9.2 `
      seaborn==0.13.2
    # re-check
    $pyRecheck = @"
import importlib, sys
mods = ["joblib","sklearn","numpy","scipy","pandas","matplotlib","seaborn"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("STILL_MISSING:", ",".join(missing))
    sys.exit(1)
print("OK")
"@
    $pyRecheck | & python -
    if ($LASTEXITCODE -ne 0) {
      Err 'ML stack install failed.'
      Pause
      exit 1
    }
  }
  Ok 'ML stack OK.'
}

# ---- Workflows ----
function Get-LeagueData {
  Clear-Host
  Ensure-BasePackages
  Info 'fetching league data...'
  & python -m src.fetch_ligor
  if ($LASTEXITCODE -eq 0) { Ok 'data fetched.' } else { Err 'fetch failed.' }
  Pause
}

function Invoke-ModelTraining {
  Clear-Host
  Ensure-BasePackages
  Ensure-MLStack
  Info 'training models...'
  & python -m src.train
  if ($LASTEXITCODE -eq 0) { Ok 'training done.' } else { Err 'training failed.' }
  Pause
}

function Start-StreamlitApp {
  Clear-Host
  Ensure-BasePackages
  Info 'starting streamlit app...'
  Write-Host 'URL: http://localhost:8501' -ForegroundColor DarkGray
  Start-Process 'chrome' 'http://localhost:8501' -ErrorAction SilentlyContinue | Out-Null
  & python -m streamlit run 'src\app.py' --server.port=8501
}

function Install-Requirements {
  Clear-Host
  Info 'upgrading pip toolchain...'
  & python -m pip install --upgrade pip setuptools wheel
  if (Test-Path '.\requirements.txt') {
    Info 'installing requirements.txt...'
    & python -m pip install -r requirements.txt
  } else {
    Warn 'no requirements.txt found - skipping.'
  }
  Pause
}

# ---- Menu loop ----
while ($true) {
  Clear-Host
  Write-Host ''
  Write-Host '[INFO] FOOTBALL-AI - MAIN MENU' -ForegroundColor Cyan
  Write-Host '==========================================' -ForegroundColor DarkGray
  Write-Host '  [1] Get league data'
  Write-Host '  [2] Invoke model training'
  Write-Host '  [3] Start Streamlit app'
  Write-Host '  [4] Install/Update requirements.txt'
  Write-Host '  [Q] Quit'
  Write-Host '==========================================' -ForegroundColor DarkGray
  Write-Host ''
  $choice = Read-Host 'Choose [1-4 or Q]'

  if ($choice -eq '1') { Get-LeagueData; continue }
  if ($choice -eq '2') { Invoke-ModelTraining; continue }
  if ($choice -eq '3') { Start-StreamlitApp; break }
  if ($choice -eq '4') { Install-Requirements; continue }
  if ($choice.ToUpperInvariant() -eq 'Q') { break }

  Write-Host 'Invalid choice. Try again.' -ForegroundColor Yellow
  Start-Sleep -Milliseconds 900
}

Ok 'bye. thanks for using Football-AI.'
