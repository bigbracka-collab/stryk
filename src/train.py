"""
src/train.py
~~~~~~~~~~~~
Tränar en modell per division (Premier, Championship, League One, etc.).
Divisionerna hålls separerade – ingen pd.concat över ligor.

Nyheter:
- CLI: välj division(er), dry-run, minsta radantal, seed, parallell-träning mellan divisioner
- Sammanfattning sparas som CSV (param/training_summary.csv) + JSON med körmetadata
- Tidsmätning och robustare validering av filnamn/säsong
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from joblib import Parallel, delayed

from src.config import DATA_DIR, MODELL_DIR, PARAM_DIR, setup_logging
from src.train_league import träna_liga

setup_logging()
log = logging.getLogger(__name__)


# -----------------------------
# Hjälpare
# -----------------------------

def _split_name(name: str) -> Optional[Tuple[str, str]]:
    """
    Förväntar sig 'england_premier_2526' -> ('england_premier', '2526').
    Returnerar None om mönstret inte matchar.
    """
    parts = name.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0], parts[1]


def hitta_divisionsfiler(min_rows: int = 1) -> Dict[str, List[Path]]:
    """
    Grupperar CSV-filer per division (t.ex. 'england_premier', 'england_championship').
    Filtrerar bort filer som inte följer namnmönstret eller är för små.
    """
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        log.warning(f"Inga CSV-filer funna i {DATA_DIR}")
        return {}

    grupper: Dict[str, List[Path]] = {}
    for fil in csv_files:
        name = fil.stem  # t.ex. "england_premier_2526"
        split = _split_name(name)
        if split is None:
            log.warning(f"Ignorerar fil med ogiltigt namn (förväntade *_YYYY): {fil.name}")
            continue

        div_name, season = split
        # Snabb storlekskontroll utan att läsa hela filen (best effort)
        try:
            # Läs bara 5 rader för att säkerställa att den inte är tom/trasig
            with fil.open("r", encoding="utf-8") as fh:
                header = fh.readline()
                body_peek = [fh.readline() for _ in range(5)]
            if all(not r.strip() for r in body_peek):
                log.warning(f"Ignorerar {fil.name}: verkar sakna rader.")
                continue
        except Exception as e:
            log.warning(f"Ignorerar {fil.name}: kunde inte läsa – {e}")
            continue

        grupper.setdefault(div_name, []).append(fil)

    # Sortera varje divisions filer på säsong (stigande)
    for div, filer in grupper.items():
        grupper[div] = sorted(filer, key=lambda p: _split_name(p.stem)[1] if _split_name(p.stem) else "0000")

    return grupper


@dataclass
class TrainResult:
    division: str
    ok: bool
    seconds: float
    params_path: Optional[str] = None
    model_home_path: Optional[str] = None
    model_away_path: Optional[str] = None
    note: Optional[str] = None
    metrics_brier: Optional[float] = None
    metrics_ece: Optional[float] = None


def _summarize_after_training(div_name: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Försöker läsa param-fil och härleda modellvägar efter träning.
    """
    param_file = PARAM_DIR / f"{div_name}_parametrar.json"
    mh = MODELL_DIR / f"{div_name}_home.pkl"
    ma = MODELL_DIR / f"{div_name}_away.pkl"

    brier = ece = None
    if param_file.exists():
        try:
            data = json.loads(param_file.read_text(encoding="utf-8"))
            brier = float(data.get("brier_score_valid") or data.get("brier_score") or 0.0)
            ece = float(data.get("ece_valid") or data.get("ece") or 0.0)
        except Exception as e:
            log.warning(f"Kunde inte läsa {param_file.name}: {e}")

    return brier, ece, (str(mh) if mh.exists() else None), (str(ma) if ma.exists() else None)


# -----------------------------
# Träning
# -----------------------------

def train_one_division(div_name: str, filer: List[Path], dry_run: bool = False) -> TrainResult:
    start = time.time()
    note = None
    try:
        if dry_run:
            note = f"Dry-run: skulle träna {div_name} på {len(filer)} fil(er)."
            log.info(note)
        else:
            log.info(f"Tränar {div_name.upper()} med {len(filer)} fil(er)...")
            log.info(f"  Filer: {[f.name for f in filer]}")
            träna_liga(div_name, [str(f) for f in filer])
            log.info(f"{div_name.upper()}: Träning klar!")

        brier, ece, mh, ma = _summarize_after_training(div_name)
        return TrainResult(
            division=div_name,
            ok=True,
            seconds=time.time() - start,
            params_path=str(PARAM_DIR / f"{div_name}_parametrar.json"),
            model_home_path=mh,
            model_away_path=ma,
            note=note,
            metrics_brier=brier,
            metrics_ece=ece,
        )
    except Exception as e:
        log.error(f"{div_name.upper()}: Träning misslyckades: {e}", exc_info=True)
        return TrainResult(
            division=div_name,
            ok=False,
            seconds=time.time() - start,
            note=str(e),
        )


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Träna Poisson-modeller per division.")
    p.add_argument("--div", nargs="*", help="Endast dessa divisioner (t.ex. england_premier england_championship).")
    p.add_argument("--exclude", nargs="*", default=[], help="Exkludera dessa divisioner.")
    p.add_argument("--dry-run", action="store_true", help="Läs och lista vad som skulle tränas, men kör inte själva träningen.")
    p.add_argument("--min-rows", type=int, default=1, help="Minsta radantal per fil (snabb heuristik, default=1).")
    p.add_argument("--parallel-divisions", type=int, default=1, help="Parallellisera mellan divisioner (OBS: train_league parallelliserar internt).")
    p.add_argument("--seed", type=int, default=42, help="Slumpfrö för reproducerbarhet (vid ev. randomitet senare).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    log.info("Startar träning av alla divisioner...")
    grupper = hitta_divisionsfiler(min_rows=args.min_rows)
    if not grupper:
        log.error("Inga giltiga divisionsfiler hittades. Avslutar.")
        return

    # Filtera divisioner enligt CLI
    divisions = sorted(grupper.keys())
    if args.div:
        wanted = set(args.div)
        divisions = [d for d in divisions if d in wanted]
        if not divisions:
            log.error(f"Inga matchande divisioner i --div: {args.div}")
            return
    if args.exclude:
        divisions = [d for d in divisions if d not in set(args.exclude)]

    log.info(f"Divisioner som kommer tränas ({len(divisions)}): {divisions}")

    # Kör träning (OBS: train_league.py använder parallellism internt – håll n_jobs lågt här)
    if args.parallel_divisions > 1 and not args.dry_run:
        results: List[TrainResult] = Parallel(n_jobs=args.parallel_divisions, verbose=10)(
            delayed(train_one_division)(div, grupper[div], args.dry_run) for div in divisions
        )
    else:
        results = [train_one_division(div, grupper[div], args.dry_run) for div in divisions]

    # Sammanställning
    ok_cnt = sum(r.ok for r in results)
    log.info(f"KLAR! {ok_cnt}/{len(results)} divisioner {'kontrollerade' if args.dry_run else 'tränade'}.")

    # Spara sammanfattning
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = PARAM_DIR / "training_summary.csv"
    summary_json = PARAM_DIR / "training_summary.json"

    # Gör enkel tabell
    rows = []
    for r in results:
        d = asdict(r)
        rows.append(d)

    try:
        # CSV
        import pandas as pd
        df = pd.DataFrame(rows)
        # Sortera trevligt
        df = df.sort_values(["ok", "division"], ascending=[False, True])
        df.to_csv(summary_csv, index=False, encoding="utf-8")
        # JSON
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dry_run": args.dry_run,
            "seed": args.seed,
            "results": rows,
        }
        summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"Sammanfattning sparad: {summary_csv.name}, {summary_json.name}")
    except Exception as e:
        log.warning(f"Kunde inte skriva sammanfattning: {e}")


if __name__ == "__main__":
    main()
