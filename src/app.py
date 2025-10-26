import os
import sys
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

# Lägg till projektroten i sys.path så att config.py hittas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PARAM_DIR, DATA_DIR
from predict import förutsäg_match

st.set_page_config(page_title="Fotbollsprognos", layout="centered")
st.title("⚽ Matchprognos")

# --- Hämta tränade ligor ---
ligor = [
    f.replace("_parametrar.json", "")
    for f in os.listdir(PARAM_DIR)
    if f.endswith("_parametrar.json")
]

if not ligor:
    st.warning("Ingen tränad liga hittades.")
    st.stop()

val_liga = st.selectbox("Välj liga", sorted(ligor))

# --- Hämta matchdata ---
csv_filer = [f for f in os.listdir(DATA_DIR) if f.startswith(val_liga) and f.endswith(".csv")]
if not csv_filer:
    st.error(f"Ingen matchdata hittades för {val_liga}.")
    st.stop()

df_lista = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in csv_filer]
df = pd.concat(df_lista, ignore_index=True)

# --- Välj lag ---
alla_lag = sorted(set(df["HomeTeam"].dropna().unique()) | set(df["AwayTeam"].dropna().unique()))
home_team = st.selectbox("🏠 Hemmalag", alla_lag)
away_team = st.selectbox("🛫 Bortalag", [lag for lag in alla_lag if lag != home_team])

# --- Kör prognos ---
if st.button("🔮 Förutsäg match"):
    resultat, sannolikheter, matrix, topp3, info = förutsäg_match(df, home_team, away_team, val_liga)

    st.subheader("📊 Resultat")
    st.write(resultat)

    st.subheader("📈 1X2-sannolikheter (Skellam)")
    st.write(f"**Vinst hemma:** {sannolikheter['H']:.1%}")
    st.write(f"**Oavgjort:** {sannolikheter['D']:.1%}")
    st.write(f"**Vinst borta:** {sannolikheter['A']:.1%}")

    st.subheader("🎯 Toppresultat (från målmatris)")
    for res, prob in topp3:
        st.write(f"{res}: {prob:.2%}")

    # --- Heatmap för målmatris ---
    st.subheader("📊 Målmatris")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=False, cmap="Blues", cbar=True, ax=ax)
    ax.set_xlabel("Bortamål")
    ax.set_ylabel("Hemmmål")
    st.pyplot(fig)

    st.markdown("---")
    st.caption(
        f"Säsong: {info.get('season','?')} | "
        f"Form: {home_team} {info['form_home']:.2f} – {info['form_away']:.2f} {away_team}"
    )

    # --- Senaste 5 matcher för båda lagen ---
    st.subheader("📅 Senaste 5 matcher")

    def senaste_matcher(df, lag, n=5):
        matcher = df[(df["HomeTeam"] == lag) | (df["AwayTeam"] == lag)].copy()
        if "Date" in matcher.columns:
            matcher["Date"] = pd.to_datetime(matcher["Date"], errors="coerce", dayfirst=True)
            matcher = matcher.drop_duplicates(
                subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
                keep="last"
            ).sort_values("Date", ascending=False)
        else:
            matcher = matcher.drop_duplicates(
                subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"],
                keep="last"
            ).sort_index(ascending=False)
        matcher = matcher.head(n)
        rows = []
        for _, rad in matcher.iterrows():
            datum = rad["Date"].strftime("%Y-%m-%d") if "Date" in rad and pd.notna(rad["Date"]) else "?"
            hem, borta = rad["HomeTeam"], rad["AwayTeam"]
            res = f"{rad['FTHG']}–{rad['FTAG']}"
            rows.append(f"{datum}: {hem} {res} {borta}")
        return rows

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{home_team}**")
        for r in senaste_matcher(df, home_team):
            st.write(r)
    with col2:
        st.markdown(f"**{away_team}**")
        for r in senaste_matcher(df, away_team):
            st.write(r)

    # --- Head-to-Head historik ---
    st.subheader("🤝 Head-to-Head")
    h2h = df[((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) |
             ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))].copy()
    if "Date" in h2h.columns:
        h2h["Date"] = pd.to_datetime(h2h["Date"], errors="coerce", dayfirst=True)
        h2h = h2h.sort_values("Date", ascending=False)
    h2h = h2h.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last")
    if not h2h.empty:
        st.write(h2h[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam"]].head(10))
    else:
        st.info("Inga inbördes möten hittades i datan.")

    # --- Kalibreringsgraf ---
    st.subheader("📐 Kalibreringskurva (modellens sannolikheter vs verkliga utfall)")
    if "Season" in df.columns:
        senaste_säsong = df["Season"].iloc[-1]
        df_eval = df[df["Season"] == senaste_säsong].copy()
        preds = []
        for _, rad in df_eval.iterrows():
            try:
                _, sann, _, _, _ = förutsäg_match(df_eval, rad["HomeTeam"], rad["AwayTeam"], val_liga)
                preds.append(sann)
            except Exception:
                preds.append({"H": np.nan, "D": np.nan, "A": np.nan})
        df_eval[["pH", "pD", "pA"]] = pd.DataFrame(preds)
        df_eval["outcome"] = np.where(
            df_eval["FTHG"] > df_eval["FTAG"], "H",
            np.where(df_eval["FTHG"] == df_eval["FTAG"], "D", "A")
        )
        bins = np.linspace(0, 1, 11)
        kalibrering = []
        for col in ["H", "D", "A"]:
            df_eval[f"bin_{col}"] = pd.cut(df_eval[f"p{col}"], bins)
            grp = df_eval.groupby(f"bin_{col}").apply(
                lambda g: pd.Series({
                    "mean_pred": g[f"p{col}"].mean(),
                    "emp_freq": (g["outcome"] == col).mean(),
                    "count": len(g)
                })
            )
            grp["resultat"] = col
            kalibrering.append(grp)
        kalibrering = pd.concat(kalibrering)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        for col, data in kalibrering.groupby("resultat"):
            ax2.plot(data["mean_pred"], data["emp_freq"], "o-", label=col)
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.7)
        ax2.set_xlabel("Förutsagd sannolikhet")
        ax2.set_ylabel("Faktisk frekvens")
        ax2.set_title(f"Kalibreringskurva – {val_liga} {senaste_säsong}")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Ingen säsongsinformation tillgänglig för kalibrering.")

# --- Visa historik för Brier score och ECE ---
st.header("📈 Modellens historik")
loggfil = os.path.join(PARAM_DIR, "brier_log.csv")
if os.path.exists(loggfil):
    df_logg = pd.read_csv(loggfil, parse_dates=["run"])
    df_liga = df_logg[df_logg["liga"] == val_liga].copy()
    if not df_liga.empty:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
