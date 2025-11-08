"""
src/fetch_ligor.py
~~~~~~~~~~~~~~~~~~
Hämtar data för flera ligor och säsonger från football-data.co.uk
Använder src.hamta_data -> sparar i data/{namn}_{säsong}.csv

Exempel:
    python -m src.fetch_ligor --alla-england --säsonger 2526 2425 2324
    python -m src.fetch_ligor --ligor E0=england_premier E1=england_championship --säsonger 2526 --overwrite
    python -m src.fetch_ligor --ligor E0 E1 E2 --auto-namn --säsonger 2526
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Säkerställ projektimporter
try:
    from src.hamta_data import hamta_data
except Exception as e:
    print("Kunde inte importera 'src.hamta_data'. Kontrollera din projektstruktur.", file=sys.stderr)
    raise

# =====================
# LOGGNING
# =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("fetch_ligor")


# =====================
# DEFAULT-KONFIG
# =====================

# Ange säsonger (format: "2324" = 2023–2024)
DEFAULT_SÄSONGER: List[str] = ["2526", "2425", "2324"]

# Ange ligor (kod => namn). Om du kör med --auto-namn räcker det med koderna.
DEFAULT_LIGOR: Dict[str, str] = {
    "E0": "england_premier",       # Premier League
    "E1": "england_championship",  # Championship
    "E2": "england_league1",       # League One
    # "E3": "england_league2",
    # "SP1": "spain_laliga",
    # "I1": "italy_seriea",
    # "D1": "germany_bundesliga",
    # "F1": "france_ligue1",
}

AUTO_NAME_MAP: Dict[str, str] = {
    "E0": "england_premier",
    "E1": "england_championship",
    "E2": "england_league1",
    "E3": "england_league2",
    "SC0": "scotland_premier",
    "D1": "germany_bundesliga",
    "I1": "italy_seriea",
    "SP1": "spain_laliga",
    "F1": "france_ligue1",
}

# =====================
# DATATYPER
# =====================

@dataclass
class FetchTask:
    ligakod: str
    säsong: str
    namn: str
    overwrite: bool = False
    attempts: int = 3        # max antal försök
    backoff_sec: float = 2.0 # bas-backoff

@dataclass
class FetchResult:
    task: FetchTask
    ok: bool
    filepath: Optional[Path]
    rows: Optional[int]
    error: Optional[str]


# =====================
# KÄRNLOGIK
# =====================

def _run_one(task: FetchTask) -> FetchResult:
    """Kör ett hämt-jobb med retry + exponentiell backoff."""
    for attempt in range(1, task.attempts + 1):
        try:
            path = hamta_data(
                ligakod=task.ligakod,
                säsong=task.säsong,
                namn=task.namn,
                overwrite=task.overwrite
            )
            if path and path.exists():
                try:
                    import pandas as pd
                    rows = len(pd.read_csv(path))
                except Exception:
                    rows = None
                return FetchResult(task, True, path, rows, None)
            else:
                err = "Ingen fil skapades."
        except Exception as e:
            err = str(e)

        # Misslyckat försök → ev. backoff
        if attempt < task.attempts:
            wait = task.backoff_sec * (2 ** (attempt - 1))
            log.warning(f"{task.ligakod} {task.säsong}: försök {attempt}/{task.attempts} misslyckades ({err}). Försöker igen om {wait:.1f}s...")
            time.sleep(wait)
        else:
            log.error(f"{task.ligakod} {task.säsong}: misslyckades efter {task.attempts} försök. ({err})")
            return FetchResult(task, False, None, None, err)

    # Borde inte nås
    return FetchResult(task, False, None, None, "Okänt fel")


def _build_tasks(
    säsonger: List[str],
    ligor: Dict[str, str] | List[str],
    overwrite: bool,
    auto_namn: bool
) -> List[FetchTask]:
    tasks: List[FetchTask] = []

    # Om ligor är lista med koder (utan namn)
    if isinstance(ligor, list):
        for kod in ligor:
            namn = AUTO_NAME_MAP.get(kod.upper(), kod.lower()) if auto_namn else kod.lower()
            for s in säsonger:
                tasks.append(FetchTask(kod.upper(), s, namn, overwrite=overwrite))
        return tasks

    # Om ligor är dict med kod => namn
    for kod, namn in ligor.items():
        nm = (AUTO_NAME_MAP.get(kod.upper(), namn)) if auto_namn else namn
        for s in säsonger:
            tasks.append(FetchTask(kod.upper(), s, nm, overwrite=overwrite))
    return tasks


def fetch_alla_ligor(
    säsonger: List[str] = DEFAULT_SÄSONGER,
    ligor: Dict[str, str] | List[str] = DEFAULT_LIGOR,
    overwrite: bool = False,
    max_workers: int = 3,
    auto_namn: bool = False
) -> List[FetchResult]:
    """
    Hämtar data för alla angivna ligor och säsonger.
    Parallelliserar försiktigt (default 3 trådar).
    """
    tasks = _build_tasks(säsonger, ligor, overwrite, auto_namn)
    tot = len(tasks)
    log.info(f"Startar hämtning av {tot} filer... (workers={max_workers})")

    results: List[FetchResult] = []
    if tot == 0:
        log.warning("Inga jobb att köra.")
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        fut_map = {exe.submit(_run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(fut_map):
            res = fut.result()
            results.append(res)
            done += 1
            if res.ok:
                msg = f"KLAR {done}/{tot}: {res.task.namn}_{res.task.säsong}.csv"
                if res.rows is not None:
                    msg += f" ({res.rows} rader)"
                log.info(msg)
            else:
                log.error(f"FEL  {done}/{tot}: {res.task.ligakod} {res.task.säsong} – {res.error}")

    ok = sum(1 for r in results if r.ok)
    log.info(f"SLUT: {ok}/{tot} filer hämtade med framgång.")
    return results


# =====================
# CLI
# =====================

def _parse_cli() -> Tuple[List[str], Dict[str, str] | List[str], bool, int, bool]:
    import argparse

    p = argparse.ArgumentParser(description="Hämta fotbollsdata för flera ligor/säsonger.")
    p.add_argument("--säsonger", nargs="+", default=DEFAULT_SÄSONGER, help="Säsonger (t.ex. 2526 2425 2324)")
    p.add_argument("--ligor", nargs="+", help="Ligokoder eller 'kod=namn' (t.ex. E0=england_premier E1=england_championship)")
    p.add_argument("--alla-england", action="store_true", help="Hämta E0–E2 (lägg ev. till E3 själv).")
    p.add_argument("--overwrite", action="store_true", help="Skriv över befintliga CSV-filer.")
    p.add_argument("--workers", type=int, default=3, help="Parallella hämtningar (default 3).")
    p.add_argument("--auto-namn", action="store_true", help="Härled filnamn automatiskt från kod (E0->england_premier osv).")

    args = p.parse_args()

    # Bestäm ligor
    ligor: Dict[str, str] | List[str]
    if args.alla_england:
        ligor = ["E0", "E1", "E2"]  # lägg ev. till "E3"
    elif args.ligor:
        # Stöd både 'E0' och 'E0=england_premier'
        parsed_dict: Dict[str, str] = {}
        loose_codes: List[str] = []
        for token in args.ligor:
            if "=" in token:
                kod, namn = token.split("=", 1)
                parsed_dict[kod.strip().upper()] = namn.strip()
            else:
                loose_codes.append(token.strip().upper())
        if parsed_dict and loose_codes:
            # Blanda inte formerna; om både finns, använd auto-namn för lösa koder
            for k in loose_codes:
                parsed_dict[k] = AUTO_NAME_MAP.get(k, k.lower()) if args.auto_namn else k.lower()
            ligor = parsed_dict
        elif parsed_dict:
            ligor = parsed_dict
        else:
            ligor = loose_codes
    else:
        ligor = DEFAULT_LIGOR

    return args.säsonger, ligor, args.overwrite, max(1, args.workers), args.auto_namn


if __name__ == "__main__":
    säsonger, ligor, overwrite, workers, auto_namn = _parse_cli()
    fetch_alla_ligor(
        säsonger=säsonger,
        ligor=ligor,
        overwrite=overwrite,
        max_workers=workers,
        auto_namn=auto_namn
    )
