"""
src/config.py
~~~~~~~~~~~~~
Central konfiguration för projektet:
- Sökvägar: DATA_DIR, MODELL_DIR, PARAM_DIR, LOG_DIR
- Defaults för modellparametrar
- Loggning (setup_logging)
- Hjälpfunktioner för att läsa/spara liga-parametrar

Används av: app.py, train.py, train_league.py, predict.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------
# Sökvägar
# ---------------------------------------------------------
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELL_DIR: Path = PROJECT_ROOT / "modeller"
PARAM_DIR: Path = PROJECT_ROOT / "param"          # här sparas *_parametrar.json
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Se till att kataloger finns
for p in (DATA_DIR, MODELL_DIR, PARAM_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Defaults (kan överstyras av liga-specifika parametrar)
# ---------------------------------------------------------
DEFAULT_FORM_FACTOR: float = 1.0           # multiplicerar form-feature
DEFAULT_ZERO_INFLATION: float = 1.0        # >1.0 förstärker 0-0 i målmatrisen
DEFAULT_MAX_GOALS: int = 8                 # målmatrisens storlek (0..N)
DEFAULT_FORM_DECAY_LAMBDA: float = 0.25    # formens avklingning (högre = kortare minne)
DEFAULT_ELO_K: float = 20.0                # Elo K-faktor
DEFAULT_ELO_HOME_ADVANTAGE: float = 60.0   # Elo hemmaplansfördel (rating-poäng)

# ---------------------------------------------------------
# Loggning
# ---------------------------------------------------------
def setup_logging(level: str | int = "INFO", filename: Optional[Path] = None) -> None:
    """
    Initiera enkel fil + konsol-loggning.
    - level: 'DEBUG'/'INFO'/... eller logging.INFO
    - filename: om None -> logs/pipeline.log
    """
    log_level = logging.getLevelName(level) if isinstance(level, str) else level
    logfile = filename or (LOG_DIR / "pipeline.log")

    # Undvik dubbla handlers vid upprepade anrop
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Konsol
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt, datefmt))

    # Fil
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))

    root.setLevel(log_level)
    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        f"Loggning initierad (level={logging.getLevelName(log_level)}, file='{logfile}')"
    )

# ---------------------------------------------------------
# JSON-hjälpare
# ---------------------------------------------------------
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger(__name__).warning(f"Kunde inte läsa JSON {path.name}: {e}")
    return None

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Kunde inte skriva JSON {path.name}: {e}")

# ---------------------------------------------------------
# Liga-parametrar
# ---------------------------------------------------------
def get_league_params(league_key: str) -> Dict[str, Any]:
    """
    Läser param/{league_key}_parametrar.json om den finns,
    annars returnerar rimliga defaults.

    Example league_key: "england_premier" eller "england_championship"
    """
    path = PARAM_DIR / f"{league_key}_parametrar.json"
    data = _read_json(path) or {}

    # Fall back till defaults om fält saknas:
    merged = {
        "form_factor": data.get("form_factor", DEFAULT_FORM_FACTOR),
        "zero_inflation": data.get("zero_inflation", DEFAULT_ZERO_INFLATION),
        "max_goals": data.get("max_goals", DEFAULT_MAX_GOALS),
        # Träningsfynd (kan saknas om ej tränat än)
        "alpha": data.get("alpha"),
        "lambda": data.get("lambda", DEFAULT_FORM_DECAY_LAMBDA),
        "K": data.get("K", DEFAULT_ELO_K),
        "H": data.get("H", DEFAULT_ELO_HOME_ADVANTAGE),
        # Ev. metriker från träning
        "brier_score": data.get("brier_score") or data.get("brier_score_valid"),
        "ece": data.get("ece") or data.get("ece_valid"),
        "liga": data.get("liga", league_key),
        "notes": data.get("notes"),
    }
    return merged

# ---------------------------------------------------------
# (Valfria) små hjälpare som kan vara praktiska
# ---------------------------------------------------------
def list_divisions_from_params() -> list[str]:
    """Lista divisioner baserat på filer i PARAM_DIR."""
    return sorted([p.stem.replace("_parametrar", "") for p in PARAM_DIR.glob("*_parametrar.json")])

def list_divisions_from_data() -> list[str]:
    """Lista divisioner baserat på CSV i DATA_DIR."""
    divs = set()
    for f in DATA_DIR.glob("*.csv"):
        name = f.stem  # ex: england_premier_2526
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            divs.add(parts[0])
    return sorted(divs)
