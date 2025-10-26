import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss

# Lägg till projektroten i sys.path så att config.py hittas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATA_DIR, MODELL_DIR, PARAM_DIR, DEFAULT_FORM_FACTOR, DEFAULT_ZERO_INFLATION, DEFAULT_MAX_GOALS


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


def träna_liga(liganamn, csv_filer):
    os.makedirs(MODELL_DIR, exist_ok=True)
    os.makedirs(PARAM_DIR, exist_ok=True)

    # --- Läs och slå ihop alla säsongsfiler ---
    df_lista = []
    for fil in csv_filer:
        path = os.path.join(DATA_DIR, fil)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if {"FTHG", "FTAG", "HomeTeam", "AwayTeam"}.issubset(df.columns):
            # extrahera säsong från filnamn, ex: E0_2324.csv -> 2324
            season = fil.split("_")[1].replace(".csv", "")
            df["Season"] = season
            df_lista.append(df.dropna(subset=["FTHG", "FTAG"]))
        else:
            print(f"⚠️ Hoppar över {fil} – saknar nödvändiga kolumner.")

    if not df_lista:
        print(f"❌ Ingen giltig data för {liganamn}")
        return None

    df = pd.concat(df_lista, ignore_index=True)

    # --- Bygg features ---
    # Dummyvariabler för lag och säsong
    X_home = pd.get_dummies(df[["HomeTeam", "Season"]], drop_first=True)
    X_away = pd.get_dummies(df[["AwayTeam", "Season"]], drop_first=True)

    # Hemmaplansfördel
    X_home["home_advantage"] = 1
    X_away["home_advantage"] = 0

    # Form – beräkna en gång per lag och slå upp
    alla_lag = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    lag_form = {lag: beräkna_form(df, lag) for lag in alla_lag}

    df["form_home"] = df["HomeTeam"].map(lag_form)
    df["form_away"] = df["AwayTeam"].map(lag_form)

    X_home = pd.concat([X_home, df["form_home"].rename("form_home")], axis=1)
    X_away = pd.concat([X_away, df["form_away"].rename("form_away")], axis=1)

    # Målvariabler
    y_home = df["FTHG"]
    y_away = df["FTAG"]

    # --- Träna modeller ---
    modell_home = PoissonRegressor(alpha=0.1, max_iter=300).fit(X_home, y_home)
    modell_away = PoissonRegressor(alpha=0.1, max_iter=300).fit(X_away, y_away)

    # --- Spara modeller ---
    joblib.dump(modell_home, os.path.join(MODELL_DIR, f"{liganamn}_home.pkl"))
    joblib.dump(modell_away, os.path.join(MODELL_DIR, f"{liganamn}_away.pkl"))

    # --- Bygg målmatris för sannolikheter ---
    exp_home = modell_home.predict(X_home).mean()
    exp_away = modell_away.predict(X_away).mean()

    max_goals = DEFAULT_MAX_GOALS
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_home = np.exp(-exp_home) * exp_home**i / math.factorial(i)
            p_away = np.exp(-exp_away) * exp_away**j / math.factorial(j)
            matrix[i, j] = p_home * p_away

    if DEFAULT_ZERO_INFLATION:
        matrix[0, 0] *= 1.1
        matrix /= matrix.sum()

    prob_H = float(np.sum(np.tril(matrix, -1)))
    prob_D = float(np.sum(np.diag(matrix)))
    prob_A = float(np.sum(np.triu(matrix, 1)))

    # --- Brier score ---
    outcomes = []
    for h, a in zip(df["FTHG"], df["FTAG"]):
        if h > a:
            outcomes.append([1, 0, 0])
        elif h == a:
            outcomes.append([0, 1, 0])
        else:
            outcomes.append([0, 0, 1])
    outcomes = np.array(outcomes)

    probs = np.tile([prob_H, prob_D, prob_A], (len(outcomes), 1))
    brier = (
        brier_score_loss(outcomes[:, 0], probs[:, 0]) +
        brier_score_loss(outcomes[:, 1], probs[:, 1]) +
        brier_score_loss(outcomes[:, 2], probs[:, 2])
    ) / 3

    # --- Spara parametrar ---
    paramfil = os.path.join(PARAM_DIR, f"{liganamn}_parametrar.json")
    with open(paramfil, "w", encoding="utf-8") as f:
        json.dump({
            "brier_score": round(brier, 4),
            "form_factor": DEFAULT_FORM_FACTOR,
            "zero_inflation": DEFAULT_ZERO_INFLATION
        }, f, ensure_ascii=False, indent=2)

    # --- Skapa träffgraf ---
    plt.figure(figsize=(6, 4))
    plt.bar(["H", "D", "A"], [prob_H, prob_D, prob_A], color="skyblue")
    plt.title(f"{liganamn} – genomsnittliga sannolikheter")
    plt.ylabel("Sannolikhet")
    plt.tight_layout()
    plt.savefig(os.path.join(PARAM_DIR, f"{liganamn}_träffgraf.png"))
    plt.close()

    print(f"✅ Klar: {liganamn} – Brier score: {brier:.4f}")
    return {
        "liga": liganamn,
        "brier_score": round(brier, 4),
        "form_factor": DEFAULT_FORM_FACTOR,
        "zero_inflation": DEFAULT_ZERO_INFLATION
    }
