"""
src/train_league.py
~~~~~~~~~~~~~~~~~~~
Tränar Poisson-modeller (hemmamål/bortamål) per division – utan framtidsläckor.

- Pre-match Elo (ingen läcka)
- Pre-match form (ingen läcka)
- Tidsbaserad validering (senaste säsong eller sista 20%)
- Cache av Elo(K,H) och Form(λ)
- Kompatibel OneHotEncoder (sparse_output för ny sklearn, fallback till sparse)

Sparar:
- modeller/{liga}_home.pkl
- modeller/{liga}_away.pkl
- param/{liga}_parametrar.json
"""

from __future__ import annotations

import json
import logging
import math
import inspect
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from itertools import product

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss
from scipy.stats import skellam

# IMPORT från config.py
from src.config import (
    MODELL_DIR, PARAM_DIR,
    DEFAULT_FORM_FACTOR, DEFAULT_ZERO_INFLATION, DEFAULT_MAX_GOALS,
)

log = logging.getLogger(__name__)


# =====================
# Hjälpfunktioner
# =====================

def safe_log(x: pd.Series) -> pd.Series:
    """Stabil log-transform av odds-serier."""
    return np.log(np.clip(pd.to_numeric(x, errors="coerce"), 1.01, None))

def implied_prob(over_odds: np.ndarray, under_odds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalisera over/under-odds till sannolikheter (p_over, p_under)."""
    over_odds = np.clip(pd.to_numeric(over_odds, errors="coerce"), 1.01, None)
    under_odds = np.clip(pd.to_numeric(under_odds, errors="coerce"), 1.01, None)
    inv_sum = (1.0 / over_odds) + (1.0 / under_odds)
    inv_sum = np.clip(inv_sum, 1e-8, None)
    p_over = (1.0 / over_odds) / inv_sum
    p_under = (1.0 / under_odds) / inv_sum
    return p_over, p_under

def calc_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error för 1X2-sannolikheter."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(3):
        p = y_prob[:, i]
        true = (y_true == i).astype(int)
        binids = np.digitize(p, edges) - 1
        for b in range(bins):
            mask = (binids == b)
            if mask.any():
                ece += abs(p[mask].mean() - true[mask].mean()) * (mask.sum() / len(y_true))
    return float(ece)

def _model_1x2_probs(mu_h: float, mu_a: float) -> Tuple[float, float, float]:
    """1X2 via Skellam utifrån förväntade mål (mu_h, mu_a)."""
    pD = skellam.pmf(0, mu1=mu_h, mu2=mu_a)
    pH = 1.0 - skellam.cdf(0, mu1=mu_h, mu2=mu_a)
    pA = skellam.cdf(-1, mu1=mu_h, mu2=mu_a)
    s = pH + pD + pA
    if not np.isfinite(s) or s <= 0:
        return 1/3, 1/3, 1/3
    return pH / s, pD / s, pA / s

def _ohe_kwargs() -> Dict[str, Any]:
    """Bygg kwargs för OneHotEncoder som funkar i både ny och gammal sklearn."""
    sig = inspect.signature(OneHotEncoder.__init__)
    if "sparse_output" in sig.parameters:
        return {"handle_unknown": "ignore", "sparse_output": False}
    # äldre scikit-learn
    return {"handle_unknown": "ignore", "sparse": False}

def bygg_pipeline(cat_features: List[str], num_features: List[str], alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(**_ohe_kwargs()), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    model = PoissonRegressor(alpha=alpha, max_iter=5000, fit_intercept=True)
    return Pipeline([("prep", preprocessor), ("model", model)])


# =====================
# Läckfria features
# =====================

def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställ en giltig Date-kolumn för tidsordning."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    else:
        df["Date"] = pd.to_datetime(np.arange(len(df)), unit="D", origin="1970-01-01")
    return df

def compute_pre_match_elo(df: pd.DataFrame, K: float, H: float) -> Tuple[pd.Series, pd.Series]:
    """
    Returnerar PRE-match Elo (home, away) för varje rad – dvs Elo som gällde
    precis innan matchen spelades. Ingen framtidsläcka.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True))
    elo = {t: 1500.0 for t in teams}

    pre_h = np.empty(len(df))
    pre_a = np.empty(len(df))

    for i, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        Rh, Ra = elo.get(h, 1500.0), elo.get(a, 1500.0)

        # Spara pre-match elo
        pre_h[i], pre_a[i] = Rh, Ra

        # Expected score (home) med hemmaplansfördel H
        Eh = 1.0 / (1.0 + 10 ** (-(Rh + H - Ra) / 400.0))

        # Utfall i hemmalagsperspektiv
        if row["FTHG"] > row["FTAG"]:
            Sh = 1.0
        elif row["FTHG"] == row["FTAG"]:
            Sh = 0.5
        else:
            Sh = 0.0
        Sa = 1.0 - Sh

        # Uppdatera POST-match
        elo[h] = Rh + K * (Sh - Eh)
        elo[a] = Ra + K * (Sa - (1 - Eh))

    ser_h = pd.Series(pre_h, index=df.index)
    ser_a = pd.Series(pre_a, index=df.index)
    return ser_h, ser_a

def compute_pre_match_form(df: pd.DataFrame, lam: float = 0.25) -> Tuple[pd.Series, pd.Series]:
    """
    Pre-match form per lag via exponentiellt medel på poäng (3/1/0),
    beräknat till och med föregående match (shiftad). Normaliseras ungefär till [0,1] via /3.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Poäng i hemmalagsperspektiv
    res_home = np.where(df["FTHG"] > df["FTAG"], 3.0, np.where(df["FTHG"] == df["FTAG"], 1.0, 0.0))
    res_away = 3.0 - res_home  # spegling

    long = pd.concat([
        pd.DataFrame({"Team": df["HomeTeam"], "Date": df["Date"], "points": res_home}),
        pd.DataFrame({"Team": df["AwayTeam"], "Date": df["Date"], "points": res_away}),
    ], ignore_index=True).sort_values(["Team", "Date"])

    # EWM + shift(1) för att utesluta aktuell match → PRE-match form
    alpha = 1 - np.exp(-float(lam))
    long["form"] = (
        long.groupby("Team", group_keys=False)["points"]
            .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean().shift(1))
    )

    # Plocka tillbaka till matchrader (sista kända form före matchdatum)
    key = long.groupby(["Team", "Date"])["form"].last()

    pre_form_h = []
    pre_form_a = []
    for i, row in df.iterrows():
        pre_form_h.append(key.get((row["HomeTeam"], row["Date"]), np.nan))
        pre_form_a.append(key.get((row["AwayTeam"], row["Date"]), np.nan))

    pre_form_h = np.nan_to_num(np.array(pre_form_h) / 3.0, nan=0.0)
    pre_form_a = np.nan_to_num(np.array(pre_form_a) / 3.0, nan=0.0)

    return pd.Series(pre_form_h, index=df.index), pd.Series(pre_form_a, index=df.index)


# =====================
# Utvärdering (train/valid)
# =====================

def evaluate_pair(
    alpha: float,
    df: pd.DataFrame,
    outcomes: np.ndarray,
    elo_h: pd.Series, elo_a: pd.Series,
    form_h: pd.Series, form_a: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> Tuple[float, Dict[str, float], Tuple[Pipeline, Pipeline], Dict[str, float]]:
    """
    Träna två Poisson-modeller (hem/bortamål) och utvärdera på validering.
    Returnerar (score=brier+ece, cfg, (home_pipe, away_pipe), metrics).
    """
    df_loc = df.copy()
    df_loc["elo_home"] = elo_h
    df_loc["elo_away"] = elo_a
    df_loc["form_decay_home"] = form_h
    df_loc["form_decay_away"] = form_a

    # Featurematriser (matchar predict/app pipelines)
    X_home = df_loc[[
        "HomeTeam", "Season", "iv",
        "form_decay_home", "elo_home", "elo_away",
        "odds_home", "odds_draw", "odds_away",
        "p_over25", "p_under25"
    ]].copy()
    X_home["home_advantage"] = 1
    y_home = df_loc["FTHG"].astype(float)

    X_away = df_loc[[
        "AwayTeam", "Season", "iv",
        "form_decay_away", "elo_home", "elo_away",
        "odds_home", "odds_draw", "odds_away",
        "p_over25", "p_under25"
    ]].copy()
    X_away["home_advantage"] = 0
    y_away = df_loc["FTAG"].astype(float)

    # Split
    Xh_tr, Xh_va = X_home.iloc[train_idx], X_home.iloc[valid_idx]
    yh_tr, yh_va = y_home.iloc[train_idx], y_home.iloc[valid_idx]
    Xa_tr, Xa_va = X_away.iloc[train_idx], X_away.iloc[valid_idx]
    ya_tr, ya_va = y_away.iloc[train_idx], y_away.iloc[valid_idx]

    pipe_home = bygg_pipeline(
        cat_features=["HomeTeam", "Season", "iv"],
        num_features=["form_decay_home", "elo_home", "elo_away",
                      "odds_home", "odds_draw", "odds_away",
                      "p_over25", "p_under25", "home_advantage"],
        alpha=alpha
    ).fit(Xh_tr, yh_tr)

    pipe_away = bygg_pipeline(
        cat_features=["AwayTeam", "Season", "iv"],
        num_features=["form_decay_away", "elo_home", "elo_away",
                      "odds_home", "odds_draw", "odds_away",
                      "p_over25", "p_under25", "home_advantage"],
        alpha=alpha
    ).fit(Xa_tr, ya_tr)

    mu_h = pipe_home.predict(Xh_va)
    mu_a = pipe_away.predict(Xa_va)
    probs = np.array([_model_1x2_probs(mh, ma) for mh, ma in zip(mu_h, mu_a)])

    # Metriker
    y_true = outcomes[valid_idx]  # 0(H),1(D),2(A)
    y_bin = np.eye(3)[y_true]
    brier = float(np.mean([brier_score_loss(y_bin[:, i], probs[:, i]) for i in range(3)]))
    ece = float(calc_ece(y_true, probs))
    score = brier + ece

    return score, {"alpha": alpha}, (pipe_home, pipe_away), {"brier": brier, "ece": ece}


# =====================
# Huvudträning – per division
# =====================

def träna_liga(liganamn: str, csv_filer: List[str]) -> Optional[Dict[str, Any]]:
    """
    Tränar en modell för en specifik division.
    csv_filer: lista med FULLA sökvägar till CSV-filer för denna division.
    """
    MODELL_DIR.mkdir(parents=True, exist_ok=True)
    PARAM_DIR.mkdir(parents=True, exist_ok=True)

    # --- Läs in & harmonisera ---
    df_list: List[pd.DataFrame] = []
    for fil_path in csv_filer:
        path = Path(fil_path)
        if not path.exists():
            log.warning(f"Fil saknas: {path}")
            continue
        try:
            df = pd.read_csv(path)
            log.info(f"Läser: {path.name} ({len(df)} rader)")
        except Exception as e:
            log.error(f"Kunde inte läsa {path}: {e}")
            continue

        df = df.rename(columns={
            "Div": "iv",
            "B365H": "B365H", "B365D": "B365D", "B365A": "B365A",
            "B365>2.5": "Avg>2.5", "B365<2.5": "Avg<2.5",
            "Avg>2.5": "Avg>2.5", "Avg<2.5": "Avg<2.5",
        }, errors="ignore")

        required = {"FTHG", "FTAG", "HomeTeam", "AwayTeam"}
        if not required.issubset(df.columns):
            log.warning(f"Hoppar över {path.name}: saknar kolumner {required - set(df.columns)}")
            continue

        # Säsong från filnamn (england_premier_2526.csv → 2526)
        season = path.stem.rsplit("_", 1)[-1]
        df["Season"] = season

        df = _prepare_dates(df)
        df = df.dropna(subset=["FTHG", "FTAG"])
        df_list.append(df)

    if not df_list:
        log.error(f"Ingen data för {liganamn}")
        return None

    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)
    log.info(f"{liganamn}: {len(df)} matcher totalt")

    # --- Basfeatures (odds + OU) ---
    # Om B365H/D/A saknas i någon fil, ersätt med NaN (safe_log hanterar)
    for c in ["B365H", "B365D", "B365A"]:
        if c not in df.columns:
            df[c] = np.nan

    df["odds_home"] = safe_log(df["B365H"])
    df["odds_draw"] = safe_log(df["B365D"])
    df["odds_away"] = safe_log(df["B365A"])

    if {"Avg>2.5", "Avg<2.5"}.issubset(df.columns):
        p_over, p_under = implied_prob(df["Avg>2.5"].values, df["Avg<2.5"].values)
        df["p_over25"], df["p_under25"] = p_over, p_under
    else:
        df["p_over25"], df["p_under25"] = 0.5, 0.5

    # 1X2 utfall (0,1,2) – används för Brier/ECE
    outcomes = np.where(df["FTHG"] > df["FTAG"], 0,
                 np.where(df["FTHG"] == df["FTAG"], 1, 2)).astype(int)

    # --- Tidsbaserad validering ---
    seasons = sorted(df["Season"].astype(str).unique().tolist())
    if len(seasons) >= 2:
        last_season = seasons[-1]
        valid_idx = df.index[df["Season"].astype(str) == last_season].to_numpy()
        train_idx = df.index[df["Season"].astype(str) != last_season].to_numpy()
        log.info(f"Validerar på säsong {last_season} ({len(valid_idx)} rader), tränar på {len(train_idx)} rader.")
    else:
        split = max(1, int(0.8 * len(df)))
        train_idx = np.arange(split)
        valid_idx = np.arange(split, len(df))
        log.info(f"Validerar på sista 20% ({len(valid_idx)} rader), tränar på {len(train_idx)} rader.")

    # --- Grids ---
    alpha_grid = [0.01, 0.05, 0.1, 0.2]
    lambda_grid = [0.10, 0.25, 0.50]
    K_grid = [15, 20, 30]
    H_grid = [50, 60, 80]

    # --- Cache pre-match form/elo ---
    form_cache: Dict[float, Tuple[pd.Series, pd.Series]] = {}
    elo_cache: Dict[Tuple[int, int], Tuple[pd.Series, pd.Series]] = {}

    for lam in lambda_grid:
        f_h, f_a = compute_pre_match_form(df, lam=lam)
        form_cache[lam] = (f_h, f_a)

    for K, H in product(K_grid, H_grid):
        e_h, e_a = compute_pre_match_elo(df, K=K, H=H)
        elo_cache[(K, H)] = (e_h, e_a)

    # --- Sök bästa kombination ---
    best: Optional[Tuple[float, Dict[str, float], Tuple[Pipeline, Pipeline], Dict[str, float]]] = None

    for K, H in product(K_grid, H_grid):
        elo_h, elo_a = elo_cache[(K, H)]
        for lam in lambda_grid:
            form_h, form_a = form_cache[lam]

            # Parallellisera bara över alpha (minskar minnestryck)
            jobs = (
                delayed(evaluate_pair)(
                    alpha, df, outcomes, elo_h, elo_a, form_h, form_a, train_idx, valid_idx
                )
                for alpha in alpha_grid
            )
            results = Parallel(n_jobs=-1, verbose=0)(jobs)

            for (score, cfg, models, metrics) in results:
                if (best is None) or (score < best[0]):
                    best = (score, {**cfg, "lambda": lam, "K": K, "H": H}, models, metrics)

    if best is None:
        log.error("Hittade ingen giltig modellkombination.")
        return None

    best_score, best_cfg, best_models, best_metrics = best

    # --- Spara modeller ---
    MODELL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_models[0], MODELL_DIR / f"{liganamn}_home.pkl")
    joblib.dump(best_models[1], MODELL_DIR / f"{liganamn}_away.pkl")

    # --- Spara parametrar/rapport ---
    params = {
        "liga": liganamn,
        "alpha": float(best_cfg["alpha"]),
        "lambda": float(best_cfg["lambda"]),
        "K": int(best_cfg["K"]),
        "H": int(best_cfg["H"]),
        "brier_score_valid": round(float(best_metrics["brier"]), 5),
        "ece_valid": round(float(best_metrics["ece"]), 5),
        # ev. downstream-defaults
        "form_factor": DEFAULT_FORM_FACTOR,
        "zero_inflation": DEFAULT_ZERO_INFLATION,
        "max_goals": DEFAULT_MAX_GOALS,
        "notes": "Pre-match Elo/Form utan läckor + tidsvalidering (senaste säsong eller sista 20%).",
    }
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    (PARAM_DIR / f"{liganamn}_parametrar.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(
        f"{liganamn} klar: Brier={best_metrics['brier']:.5f}, "
        f"ECE={best_metrics['ece']:.5f}, cfg={best_cfg}"
    )
    return params
