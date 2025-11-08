"""
src/hamta_data.py
~~~~~~~~~~~~~~~~~
Hämtar fotbollsdata från football-data.co.uk
och sparar i data/{namn}_{säsong}.csv

Exempel:
    python -m src.hamta_data --ligor E0,E1,E2 --säsong 2425
    python -m src.hamta_data --alla-england --säsong 2526 --overwrite
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests

# =====================
# KONFIG & LOGGNING
# =====================

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("hamta_data")

# =====================
# HUVUDFUNKTION
# =====================

def hamta_data(
    ligakod: str,
    säsong: str,
    namn: Optional[str] = None,
    overwrite: bool = False,
    timeout: int = 15,
) -> Optional[Path]:
    """
    Hämtar och sparar data för en liga/säsong.
    """
    namn = namn or ligakod.lower()
    filnamn = f"{namn}_{säsong}.csv"
    filpath = DATA_DIR / filnamn

    # --- CACHE ---
    if filpath.exists() and not overwrite:
        try:
            df = pd.read_csv(filpath)
            log.info(f"🟡 Använder cache: {filnamn} ({len(df)} rader)")
            return filpath
        except Exception as e:
            log.warning(f"Kunde inte läsa cache {filnamn}: {e}")

    url = f"https://www.football-data.co.uk/mmz4281/{säsong}/{ligakod}.csv"
    log.info(f"⬇️  Hämtar {url}")

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            log.error(f"{ligakod}: HTTP {resp.status_code}")
            return None
        if len(resp.content) < 200:
            log.error(f"{ligakod}: Tomt svar från servern.")
            return None
        df = pd.read_csv(io.StringIO(resp.text))
    # alternativt binärt:
    # import io
    # df = pd.read_csv(io.BytesIO(resp.content))
    
    except Exception as e:
        log.error(f"❌ Nätverksfel ({ligakod}): {e}")
        return None

    if df.empty:
        log.error(f"{ligakod}: Ingen data (tom DataFrame).")
        return None

    # --- Rensa & harmonisera ---
    df = df.rename(columns={
        "Div": "iv",
        "B365>2.5": "Avg>2.5",
        "B365<2.5": "Avg<2.5",
    }, errors="ignore")

    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        log.error(f"{ligakod}: Saknar kolumner {required - set(df.columns)}")
        return None

    # --- Spara ---
    try:
        df.to_csv(filpath, index=False, encoding="utf-8")
        log.info(f"✅ Sparad: {filnamn} ({len(df)} rader)")
        return filpath
    except Exception as e:
        log.error(f"{ligakod}: Kunde inte spara – {e}")
        return None


# =====================
# FLERLIGAFUNKTION
# =====================

def hamta_flera(
    ligor: List[str],
    säsong: str,
    overwrite: bool = False
) -> List[Path]:
    """
    Hämtar flera ligor i följd.
    """
    saved_files: List[Path] = []
    for kod in ligor:
        name_map = {
            "E0": "england_premier",
            "E1": "england_championship",
            "E2": "england_league1",
            "E3": "england_league2",
            "SC0": "scotland_premier",
            "D1": "germany_bundesliga",
            "I1": "italy_seriea",
            "SP1": "spain_laliga",
            "F1": "france_ligue1"
        }
        namn = name_map.get(kod.upper(), kod.lower())
        res = hamta_data(kod.upper(), säsong, namn=namn, overwrite=overwrite)
        if res:
            saved_files.append(res)
        time.sleep(1.5)  # artig delay mellan requests
    log.info(f"KLAR — hämtade {len(saved_files)} filer.")
    return saved_files


# =====================
# CLI
# =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hämta fotbollsdata från football-data.co.uk")
    parser.add_argument("--ligor", type=str, help="Komma-separerade ligakoder, t.ex. E0,E1,E2")
    parser.add_argument("--alla-england", action="store_true", help="Hämta E0–E3 (Premier till League Two)")
    parser.add_argument("--säsong", required=True, type=str, help="Säsong i formatet 2324, 2425, 2526...")
    parser.add_argument("--overwrite", action="store_true", help="Skriv över befintliga filer")

    args = parser.parse_args()

    if args.alla_england:
        ligor = ["E0", "E1", "E2", "E3"]
    elif args.ligor:
        ligor = [s.strip().upper() for s in args.ligor.split(",") if s.strip()]
    else:
        log.error("Inga ligor angivna. Använd --ligor eller --alla-england")
        sys.exit(1)

    hamta_flera(ligor, args.säsong, overwrite=args.overwrite)
