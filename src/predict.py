"""
src/predict.py
~~~~~~~~~~~~~~
Förutsäger matchresultat med tränade Poisson-modeller.

- Stöd för CROSS-DIVISION: t.ex. Arsenal (Premier) vs Leeds (Championship)
- Använder hemmamodell för hemmamål och bortamodell för bortamål
- Pre-match Elo och pre-match form (ingen framtidsläcka)
- Valfri match_date för att beräkna Elo/Form till ett visst datum

Returnerar:
    text (str), probs (dict H/D/A), matrix (np.ndarray), top3 (list), extra (dict)
"""

from __future__ import annotations

import math
from math import factorial
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.stats import skellam

from src.config import (
    MODELL_DIR, PARAM_DIR, DEFAULT_MAX_GOALS, DEFAULT_FORM_FACTOR,
    DEFAULT_ZERO_INFLATION, get_league_params
)

log = logging.getLogger(__name__)


# =====================
# Hjälpfunktioner
# =====================

def safe_log(x: float) -> float:
    """Stabil log-transform av odds."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return math.log(max(float(x), 1.01))
    except Exception:
        return 0.0

def implied_prob(over: float, under: float) -> Tuple[float, float]:
    """
    Normaliserar over/under-odds till (p_over, p_under).
    Faller tillbaka till (0.5, 0.5) om ogiltiga odds.
    """
    try:
        over = float(over)
        under = float(under)
        if not (over > 1.0 and under > 1.0):
            return 0.5, 0.5
        inv_o, inv_u = 1.0 / over, 1.0 / under
        s = inv_o + inv_u
        if s <= 0:
            return 0.5, 0.5
        return inv_o / s, inv_u / s
    except Exception:
        return 0.5, 0.5

def _model_1x2_probs(mu_h: float, mu_a: float) -> Tuple[float, float, float]:
    """1X2 via Skellam (Poissonmål)."""
    pD = skellam.pmf(0, mu1=mu_h, mu2=mu_a)
    pH = 1.0 - skellam.cdf(0, mu1=mu_h, mu2=mu_a)
    pA = skellam.cdf(-1, mu1=mu_h, mu2=mu_a)
    s = pH + pD + pA
    if not np.isfinite(s) or s <= 0:
        return 1/3, 1/3, 1/3
    return pH / s, pD / s, pA / s


# =====================
# Pre-match Elo & Form (läckfria)
# =====================

def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        # Ta bort rader utan giltigt datum för tidsordning
        df = df[~df["Date"].isna()]
    else:
        df = df.copy()
        df["Date"] = pd.to_datetime(np.arange(len(df)), unit="D", origin="1970-01-01")
    return df

def compute_pre_match_elo(df_all: pd.DataFrame, K: float, H: float, cutoff: Optional[pd.Timestamp]) -> Dict[str, float]:
    """
    Pre-match Elo per lag, beräknad fram till (men exkl.) cutoff.
    Om cutoff är None → använd all tillgänglig historik.
    """
    df = _prepare_dates(df_all)
    if cutoff is not None:
        df = df[df["Date"] < cutoff]
    df = df.sort_values("Date").reset_index(drop=True)

    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True))
    elo = {t: 1500.0 for t in teams}

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        Rh, Ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
        Eh = 1.0 / (1.0 + 10 ** (-(Rh + H - Ra) / 400.0))
        # Result i hemmalagsperspektiv
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

    return elo

def compute_pre_match_form(df_all: pd.DataFrame, lam: float, cutoff: Optional[pd.Timestamp]) -> Dict[str, float]:
    """
    Pre-match form per lag via exponentiell medel på poäng (3/1/0),
    beräknad fram till (men exkl.) cutoff. Normaliseras ungefär till [0,1] genom delning med 3.
    """
    df = _prepare_dates(df_all)
    if cutoff is not None:
        df = df[df["Date"] < cutoff]
    df = df.sort_values("Date").reset_index(drop=True)

    # Poäng i hemmalagsperspektiv
    res_home = np.where(df["FTHG"] > df["FTAG"], 3.0, np.where(df["FTHG"] == df["FTAG"], 1.0, 0.0))
    res_away = 3.0 - res_home  # spegling

    long = pd.concat([
        pd.DataFrame({"Team": df["HomeTeam"], "Date": df["Date"], "points": res_home}),
        pd.DataFrame({"Team": df["AwayTeam"], "Date": df["Date"], "points": res_away}),
    ], ignore_index=True).sort_values(["Team", "Date"])

    # EWM över tid (justerad för att inte använda nuvarande match -> ingen shift behövs då cutoff exkluderar matchdagen)
    alpha = 1 - np.exp(-float(lam))
    form_series = (
        long.groupby("Team", group_keys=False)["points"]
            .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )

    long["form"] = form_series
    # Ta sista kända formen per lag före cutoff
    last_form = long.groupby("Team")["form"].last().fillna(0.0) / 3.0  # ungefär [0,1]
    return last_form.to_dict()


# =====================
# CROSS-DIVISION FÖRUTSÄGELSE
# =====================

def förutsäg_cross_division(
    df_global: pd.DataFrame,  # Alla matcher från alla divisioner (historiska)
    home_team: str,
    away_team: str,
    home_div: str,             # t.ex. "england_premier"
    away_div: str,             # t.ex. "england_championship"
    upcoming_match_row: Optional[pd.Series] = None,
    match_date: Optional[datetime] = None,  # om du vill räkna Elo/Form till ett specifikt datum
) -> Tuple[str, Dict[str, float], np.ndarray, List[Tuple[str, float]], Dict[str, Any]]:
    """
    Förutsäg match mellan lag från (ev.) olika divisioner.
    - Laddar rätt modeller per lag
    - Räknar pre-match Elo & form från df_global fram till match_date (eller slutet)
    - Bygger feature-rader som matchar training pipelines
    """
    # --- Ladda modeller ---
    try:
        model_home = joblib.load(MODELL_DIR / f"{home_div}_home.pkl")
    except Exception as e:
        raise ValueError(f"Saknar hemmamodell för {home_div}: {e}") from e

    try:
        model_away = joblib.load(MODELL_DIR / f"{away_div}_away.pkl")
    except Exception as e:
        raise ValueError(f"Saknar bortamodell för {away_div}: {e}") from e

    # --- Ladda liga-parametrar ---
    params_home = get_league_params(home_div) or {}
    params_away = get_league_params(away_div) or {}

    # Ange parametrar med bra fallback (hemmas liga styr baseline)
    form_factor = float(params_home.get("form_factor", DEFAULT_FORM_FACTOR))
    zero_inflation = float(max(params_home.get("zero_inflation", DEFAULT_ZERO_INFLATION),
                               params_away.get("zero_inflation", DEFAULT_ZERO_INFLATION)))
    lambda_decay = float(params_home.get("lambda", 0.25))
    elo_K = float(params_home.get("K", 20))
    elo_H = float(params_home.get("H", 60))
    max_goals = int(params_home.get("max_goals", DEFAULT_MAX_GOALS))

    # --- Datumhantering ---
    cutoff = pd.to_datetime(match_date) if match_date is not None else None

    # --- Pre-match Elo & Form ---
    elo_dict = compute_pre_match_elo(df_global, K=elo_K, H=elo_H, cutoff=cutoff)
    elo_home = float(elo_dict.get(home_team, 1500.0))
    elo_away = float(elo_dict.get(away_team, 1500.0))

    form_dict = compute_pre_match_form(df_global, lam=lambda_decay, cutoff=cutoff)
    form_home = float(form_dict.get(home_team, 0.0))
    form_away = float(form_dict.get(away_team, 0.0))

    # --- Odds & division/season ---
    if upcoming_match_row is not None:
        odds_h = upcoming_match_row.get("B365H", upcoming_match_row.get("AvgH", np.nan))
        odds_d = upcoming_match_row.get("B365D", upcoming_match_row.get("AvgD", np.nan))
        odds_a = upcoming_match_row.get("B365A", upcoming_match_row.get("AvgA", np.nan))
        over25 = upcoming_match_row.get("Avg>2.5", np.nan)
        under25 = upcoming_match_row.get("Avg<2.5", np.nan)
        season = str(upcoming_match_row.get("Season", "Unknown"))
        division = upcoming_match_row.get("iv", "Cross")
    else:
        # Fallback: ta senaste match som involverar något av lagen för att få rimliga odds-fält
        df_candidates = df_global[
            (df_global["HomeTeam"].isin([home_team, away_team])) |
            (df_global["AwayTeam"].isin([home_team, away_team]))
        ].copy()
        df_candidates = _prepare_dates(df_candidates).sort_values("Date")
        if df_candidates.empty:
            odds_h = odds_d = odds_a = np.nan
            over25 = under25 = np.nan
            season = "Unknown"
        else:
            last = df_candidates.iloc[-1]
            odds_h = last.get("B365H", np.nan)
            odds_d = last.get("B365D", np.nan)
            odds_a = last.get("B365A", np.nan)
            over25 = last.get("Avg>2.5", np.nan)
            under25 = last.get("Avg<2.5", np.nan)
            season = str(last.get("Season", "Unknown"))
        division = "Cross"

    p_over25, p_under25 = implied_prob(over25, under25)

    # --- Bygg feature-rader (matchar träningspipelines) ---
    base_features = {
        "Season": season,
        "iv": division,
        "form_decay_home": form_home * form_factor,
        "form_decay_away": form_away * form_factor,
        "elo_home": elo_home,
        "elo_away": elo_away,
        "odds_home": safe_log(odds_h),
        "odds_draw": safe_log(odds_d),
        "odds_away": safe_log(odds_a),
        "p_over25": p_over25,
        "p_under25": p_under25,
    }

    X_home = pd.DataFrame([{**base_features, "HomeTeam": home_team, "home_advantage": 1}])
    X_away = pd.DataFrame([{**base_features, "AwayTeam": away_team, "home_advantage": 0}])

    # --- Prediktion av förväntade mål ---
    try:
        exp_home = float(model_home.predict(X_home)[0])
    except Exception as e:
        log.exception("Fel vid prediction för hemmamål.")
        raise

    try:
        exp_away = float(model_away.predict(X_away)[0])
    except Exception as e:
        log.exception("Fel vid prediction för bortamål.")
        raise

    # Säkerställ positiva µ
    exp_home = max(exp_home, 0.05)
    exp_away = max(exp_away, 0.05)

    # --- 1X2-prob via Skellam ---
    pH, pD, pA = _model_1x2_probs(exp_home, exp_away)
    probs = {"H": float(pH), "D": float(pD), "A": float(pA)}

    # --- Målmatris ---
    max_g = int(max_goals)
    matrix = np.zeros((max_g + 1, max_g + 1), dtype=float)
    # Poisson PMF för rader/kolumner
    poisson_h = [math.exp(-exp_home) * (exp_home ** i) / factorial(i) for i in range(max_g + 1)]
    poisson_a = [math.exp(-exp_away) * (exp_away ** j) / factorial(j) for j in range(max_g + 1)]
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            matrix[i, j] = poisson_h[i] * poisson_a[j]

    if zero_inflation > 1.0:
        matrix[0, 0] *= float(zero_inflation)
        s = matrix.sum()
        if s > 0:
            matrix /= s

    # --- Topp 3 mest sannolika resultat ---
    flat = matrix.ravel()
    top_idx = np.argsort(flat)[::-1][:3]
    top3: List[Tuple[str, float]] = []
    for idx in top_idx:
        i, j = divmod(idx, max_g + 1)
        top3.append((f"{i}-{j}", float(flat[idx])))

    # --- Sammanfattningstext & extra ---
    left = f"{home_team} {exp_home:.2f}"
    right = f"{exp_away:.2f} {away_team}"
    text = f"{left} – {right} ({home_div} vs {away_div})"

    extra = {
        "exp_home": exp_home, "exp_away": exp_away,
        "form_home": form_home, "form_away": form_away,
        "elo_home": elo_home, "elo_away": elo_away,
        "odds_home": float(odds_h) if pd.notna(odds_h) else None,
        "odds_draw": float(odds_d) if pd.notna(odds_d) else None,
        "odds_away": float(odds_a) if pd.notna(odds_a) else None,
        "home_div": home_div, "away_div": away_div,
        "season": season,
        "division_tag": division,
        "match_date": cutoff.isoformat() if cutoff is not None else None,
    }

    return text, probs, matrix, top3, extra
