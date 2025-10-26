import os
import sys
import joblib
import numpy as np
import pandas as pd
import json
import math
from scipy.stats import skellam  # <-- nytt

# Lägg till projektroten i sys.path så att config.py hittas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import MODELL_DIR, PARAM_DIR, DEFAULT_MAX_GOALS, DEFAULT_FORM_FACTOR, DEFAULT_ZERO_INFLATION


def beräkna_form(df, lag, n=5):
    """Beräkna formpoäng för ett lag baserat på senaste n matcher."""
    matcher = df[(df["HomeTeam"] == lag) | (df["AwayTeam"] == lag)].tail(n)
    poäng = 0
    for _, rad in matcher.iterrows():
        if rad["HomeTeam"] == lag:
            mål, mot = rad["FTHG"], rad["FTAG"]
        else:
            mål, mot = rad["FTAG"], rad["FTHG"]
        if mål > mot:
            poäng += 3
        elif mål == mot:
            poäng += 1
    return poäng / (n * 3)


def förutsäg_match(df, home_team, away_team, liganamn):
    # --- Ladda modeller ---
    try:
        modell_home = joblib.load(os.path.join(MODELL_DIR, f"{liganamn}_home.pkl"))
        modell_away = joblib.load(os.path.join(MODELL_DIR, f"{liganamn}_away.pkl"))
    except FileNotFoundError:
        raise ValueError(f"Modeller saknas för liga '{liganamn}'")

    # --- Ladda parametrar ---
    paramfil = os.path.join(PARAM_DIR, f"{liganamn}_parametrar.json")
    if os.path.exists(paramfil):
        with open(paramfil, encoding="utf-8") as f:
            params = json.load(f)
        form_factor = params.get("form_factor", DEFAULT_FORM_FACTOR)
        zero_inflation = params.get("zero_inflation", DEFAULT_ZERO_INFLATION)
    else:
        form_factor = DEFAULT_FORM_FACTOR
        zero_inflation = DEFAULT_ZERO_INFLATION

    # --- Förberäkna form för alla lag och slå upp ---
    alla_lag = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    lag_form = {lag: beräkna_form(df, lag) for lag in alla_lag}

    form_home = lag_form.get(home_team, 0.0)
    form_away = lag_form.get(away_team, 0.0)

    # --- Hämta senaste säsongen i datan ---
    season = str(df["Season"].iloc[-1]) if "Season" in df.columns else ""

    # --- Bygg feature-matris för prediction ---
    X_home = pd.DataFrame([{
        "HomeTeam": home_team,
        "Season": season,
        "home_advantage": 1,
        "form_home": form_home * form_factor
    }])
    X_home = pd.get_dummies(X_home, drop_first=True)

    X_away = pd.DataFrame([{
        "AwayTeam": away_team,
        "Season": season,
        "home_advantage": 0,
        "form_away": form_away * form_factor
    }])
    X_away = pd.get_dummies(X_away, drop_first=True)

    # Säkerställ att kolumner matchar träningsmodellerna
    for col in modell_home.feature_names_in_:
        if col not in X_home.columns:
            X_home[col] = 0
    for col in modell_away.feature_names_in_:
        if col not in X_away.columns:
            X_away[col] = 0

    X_home = X_home[modell_home.feature_names_in_]
    X_away = X_away[modell_away.feature_names_in_]

    # --- Prediktera förväntade mål ---
    exp_home = float(modell_home.predict(X_home)[0])
    exp_away = float(modell_away.predict(X_away)[0])

    # --- Skellam för 1X2 ---
    prob_H = 1 - skellam.cdf(0, mu1=exp_home, mu2=exp_away)   # D > 0
    prob_D = skellam.pmf(0, mu1=exp_home, mu2=exp_away)       # D = 0
    prob_A = skellam.cdf(-1, mu1=exp_home, mu2=exp_away)      # D < 0

    sannolikheter = {"H": prob_H, "D": prob_D, "A": prob_A}

    # --- Målmatris (för toppresultat) ---
    max_goals = DEFAULT_MAX_GOALS
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_home = np.exp(-exp_home) * exp_home**i / math.factorial(i)
            p_away = np.exp(-exp_away) * exp_away**j / math.factorial(j)
            matrix[i, j] = p_home * p_away

    if zero_inflation:
        matrix[0, 0] *= 1.1
        matrix /= matrix.sum()

    # --- Topputfall ---
    flat = matrix.flatten()
    indices = np.argsort(flat)[::-1][:3]
    topp3 = [(f"{i}-{j}", float(flat[idx])) for idx in indices for i, j in [divmod(idx, max_goals + 1)]]

    resultat = f"Förväntade mål: {home_team} {exp_home:.2f} – {exp_away:.2f} {away_team} (säsong {season})"

    return resultat, sannolikheter, matrix, topp3, {
        "exp_home": exp_home,
        "exp_away": exp_away,
        "form_home": form_home,
        "form_away": form_away,
        "season": season
    }
