"""
src/app.py - Betting with multiple bookmakers (Bet365, Pinnacle) via The Odds API.
Features:
- Fetch odds with cooldown (last_fetch_ts)
- Choose bookmakers (bet365, pinnacle, or custom list)
- Bankroll in sidebar
- Cross-division prediction + Kelly (half Kelly)
- Bets history + CSV export
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# keep seaborn import minimal to avoid unicode issues in comments
import seaborn as sns

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PARAM_DIR, DATA_DIR, setup_logging
from src.predict import förutsäg_cross_division

# extra: model dir used in debug panel
MODELL_DIR = PROJECT_ROOT / "modeller"

setup_logging()
log = logging.getLogger(__name__)

st.set_page_config(page_title="Kelly Betting AI", layout="wide", page_icon="💸")
st.title("Kelly Betting AI - Bet365 / Pinnacle")

# ---------------------------------------------------------------------
# Odds API client (The Odds API)
# ---------------------------------------------------------------------
class Bet365API:
    def __init__(self):
        self.api_key = os.getenv("ODDS_API_KEY")
        if not self.api_key:
            st.error("Ingen API-nyckel hittades! Sätt ODDS_API_KEY i .env-filen.")
            st.stop()
        self.base_url = "https://api.the-odds-api.com/v4"
        self.region = "eu"
        self.market = "h2h"
        self.last_headers: Dict[str, str] = {}  # <- initiera för säker läsning

    def hämta_odds(self, sport: str = "soccer_epl", days_from: int = 1) -> Dict:
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": self.market,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "sport": sport,         # (dubbel men ofarlig – finns även i URL)
            "daysFrom": days_from
        }
        try:
            resp = requests.get(f"{self.base_url}/sports/{sport}/odds", params=params, timeout=12)
            # Spara rate limit-headers både i api-objektet och i session_state
            self.last_headers = dict(resp.headers)
            if "odds_headers" not in st.session_state:
                st.session_state.odds_headers = {}
            st.session_state.odds_headers = {
                "x-requests-remaining": resp.headers.get("x-requests-remaining"),
                "x-requests-used": resp.headers.get("x-requests-used"),
                "x-requests-reset": resp.headers.get("x-requests-reset")
            }
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            st.error(f"Fel vid hämtning av odds: {e}")
            return {}

    @staticmethod
    def _price_from_outcomes(outs: List[Dict[str, Any]], home: str, away: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        def price_for(name: str) -> Optional[float]:
            o = next((o for o in outs if o.get("name") == name), None)
            return float(o.get("price")) if (o and o.get("price") is not None) else None
        return price_for(home), price_for("Draw"), price_for(away)

    def parse_to_df(self, events: List[Dict[str, Any]], keep_bookmaker_col: bool = True) -> pd.DataFrame:
        if not events:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for event in events:
            home = event.get("home_team")
            away = event.get("away_team")
            if not home or not away:
                continue
            ts = event.get("commence_time")
            try:
                start_time = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
            except Exception:
                start_time = None

            for bm in event.get("bookmakers", []):
                key = bm.get("key")
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    h, d, a = self._price_from_outcomes(mkt.get("outcomes", []), home, away)
                    if h is not None and d is not None and a is not None:
                        rows.append({
                            "match": f"{home} vs {away}",
                            "home_team": home,
                            "away_team": away,
                            "start_time": start_time,
                            "bookmaker": key if keep_bookmaker_col else None,
                            "home_odds": float(h),
                            "draw_odds": float(d),
                            "away_odds": float(a),
                        })
        df = pd.DataFrame(rows)
        if keep_bookmaker_col and not df.empty:
            df["bookmaker"] = df["bookmaker"].astype(str)
        return df


# ---------------------------------------------------------------------
# Session state init (no type annotations on assignments)
# ---------------------------------------------------------------------
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 10000.0
if "bets" not in st.session_state:
    st.session_state.bets = []
if "odds_df" not in st.session_state:
    st.session_state.odds_df = pd.DataFrame()
if "best_odds_df" not in st.session_state:
    st.session_state.best_odds_df = pd.DataFrame()
if "odds_headers" not in st.session_state:
    st.session_state.odds_headers = {}
if "last_fetch_ts" not in st.session_state:
    st.session_state.last_fetch_ts = None
if "fetch_cooldown_sec" not in st.session_state:
    st.session_state.fetch_cooldown_sec = 60

api = Bet365API()

# ---------------------------------------------------------------------
# Load divisions and data
# ---------------------------------------------------------------------
@st.cache_data
def load_divisions() -> List[str]:
    return sorted([f.stem.replace("_parametrar", "") for f in PARAM_DIR.glob("*_parametrar.json")])

@st.cache_data(show_spinner=False)
def load_all_data() -> pd.DataFrame:
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        st.error("Ingen data i data/. Hämta först.")
        st.stop()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            log.warning(f"Kunde inte läsa {f.name}: {e}")
    if not dfs:
        st.error("Inga CSV-filer kunde läsas.")
        st.stop()
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"], how="any")

divisions = load_divisions()
if not divisions:
    st.error("Träna modeller först: python -m src.train")
    st.stop()

df_global = load_all_data()
teams = sorted(set(df_global["HomeTeam"]) | set(df_global["AwayTeam"]))

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Inställningar")
    st.session_state.bankroll = st.number_input(
        "Bankrulle (kr)", min_value=0.0, step=100.0, value=float(st.session_state.bankroll)
    )
    st.session_state.fetch_cooldown_sec = st.number_input(
        "Odds-cooldown (sek)", min_value=0, step=10, value=int(st.session_state.fetch_cooldown_sec)
    )
    if st.session_state.last_fetch_ts:
        dt = datetime.fromtimestamp(st.session_state.last_fetch_ts)
        st.caption("Senast hämtade odds: " + dt.strftime("%Y-%m-%d %H:%M:%S"))

# ---------------------------------------------------------------------
# UI selections
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    home_div = st.selectbox("Hemmalagets division", divisions, key="home_div")
with col2:
    away_div = st.selectbox("Bortalagets division", divisions, key="away_div")

col3, col4 = st.columns(2)
with col3:
    home_team = st.selectbox("Hemmalag", teams, key="home_team")
with col4:
    away_team = st.selectbox("Bortalag", [t for t in teams if t != home_team], key="away_team")

# ---------------------------------------------------------------------
# Fetch odds (with cooldown) and choose bookmakers
# ---------------------------------------------------------------------
colf1, colf2, colf3 = st.columns([1.2, 1.4, 2.0])
with colf1:
    sport_key = st.selectbox(
        "Sport",
        ["soccer_epl", "soccer_championship"],
        index=0 if ("premier" in home_div or "premier" in away_div) else 1,
        key="sport_key",
    )

with colf2:
    source = st.selectbox(
        "Källa (bookmakers)",
        ["Bet365", "Pinnacle", "Båda (bet365+pinnacle)", "Egen lista"],
        index=0,
    )

if source == "Bet365":
    bookmakers_param = "bet365"
elif source == "Pinnacle":
    bookmakers_param = "pinnacle"
elif source == "Båda (bet365+pinnacle)":
    bookmakers_param = "bet365,pinnacle"
else:
    defaults = ["bet365", "pinnacle", "unibet", "williamhill", "bwin", "betway", "888sport", "marathonbet"]
    chosen = st.multiselect("Välj bookmakers", defaults, default=["bet365", "pinnacle"], key="custom_books")
    bookmakers_param = ",".join(chosen) if chosen else "bet365,pinnacle"

with colf3:
    do_fetch = st.button("Uppdatera odds", use_container_width=True)

if do_fetch:
    now = time.time()
    last = st.session_state.last_fetch_ts or 0
    cooldown = int(st.session_state.fetch_cooldown_sec)
    if now - last < cooldown:
        st.warning(f"Cooldown aktiv ({cooldown - int(now - last)} s kvar).")
    else:
        events = api.hämta_odds(sport=sport_key, days_from=1)
        df_odds_all = api.parse_to_df(events, keep_bookmaker_col=True)
        st.session_state.odds_df = df_odds_all
        # säker läsning av headers även före första API-kallet
        st.session_state.odds_headers = getattr(api, "last_headers", {}) or st.session_state.get("odds_headers", {})
        st.session_state.last_fetch_ts = now
        if df_odds_all.empty:
            st.info("Inga odds returnerades just nu.")
        else:
            st.success(f"Hämtade {len(df_odds_all)} rader (match x bookmaker).")

# show quota headers if available
hdr = st.session_state.get("odds_headers", {})
used = hdr.get("x-requests-used") or hdr.get("X-Requests-Used")
remaining = hdr.get("x-requests-remaining") or hdr.get("X-Requests-Remaining")
if used or remaining:
    st.caption(f"Odds API-kvota: used={used} remaining={remaining}")

# show current odds tables
df_odds_all = st.session_state.get("odds_df", pd.DataFrame())
if df_odds_all.empty:
    st.info("Inga live odds i sessionen ännu.")
else:
    st.subheader("Odds per spelbolag")
    st.dataframe(df_odds_all.sort_values(["match", "bookmaker"]), use_container_width=True)

    best = (
        df_odds_all.groupby("match", as_index=False)
        .agg(
            home_odds=("home_odds", "max"),
            draw_odds=("draw_odds", "max"),
            away_odds=("away_odds", "max"),
        )
    )
    st.session_state.best_odds_df = best.copy()
    with st.expander("Bästa odds per match (över valda bookmakers)"):
        st.dataframe(best, use_container_width=True)

# ---------------------------------------------------------------------
# Autofill odds from selection
# ---------------------------------------------------------------------
prefill_home, prefill_draw, prefill_away = 2.0, 3.5, 3.8

if not st.session_state.get("best_odds_df", pd.DataFrame()).empty:
    src_df = st.session_state["best_odds_df"]
elif not df_odds_all.empty:
    src_df = df_odds_all.sort_values(["match", "bookmaker"]).drop_duplicates("match")
else:
    src_df = pd.DataFrame()

selected_match = None
if not src_df.empty:
    selected_match = st.selectbox("Välj match att fylla odds från", src_df["match"].tolist(), key="match_select")
    if selected_match:
        r = src_df[src_df["match"] == selected_match].iloc[0]
        prefill_home = float(r["home_odds"])
        prefill_draw = float(r["draw_odds"])
        prefill_away = float(r["away_odds"])

st.subheader("Odds (automatiskt eller manuellt)")
c1, c2, c3 = st.columns(3)
with c1:
    odds_home = st.number_input("Hemma (1)", min_value=1.01, value=prefill_home, step=0.05, key="odds_home")
with c2:
    odds_draw = st.number_input("Oavgjort (X)", min_value=1.01, value=prefill_draw, step=0.05, key="odds_draw")
with c3:
    odds_away = st.number_input("Borta (2)", min_value=1.01, value=prefill_away, step=0.05, key="odds_away")

# ---------------------------------------------------------------------
# Prediction + Kelly
# ---------------------------------------------------------------------
if st.button("Beräkna Kelly & Value", use_container_width=True):
    with st.spinner("Analyserar..."):
        upcoming = pd.Series({
            "B365H": odds_home, "B365D": odds_draw, "B365A": odds_away,
            "Avg>2.5": 1.8, "Avg<2.5": 2.0,
            "Season": "2526", "iv": "Cross",
        })

        text, probs, matrix, top3, extra = förutsäg_cross_division(
            df_global=df_global,
            home_team=home_team,
            away_team=away_team,
            home_div=home_div,
            away_div=away_div,
            upcoming_match_row=upcoming,
        )

    st.success("Analys klar!")
    st.markdown(f"### {text}")

    colp1, colp2, colp3 = st.columns(3)
    colp1.metric("Hemma", f"{probs['H']:.1%}", f"{odds_home:.2f}")
    colp2.metric("Oavgjort", f"{probs['D']:.1%}", f"{odds_draw:.2f}")
    colp3.metric("Borta", f"{probs['A']:.1%}", f"{odds_away:.2f}")

    odds_dict = {"H": odds_home, "D": odds_draw, "A": odds_away}
    value = {k: probs[k] * odds_dict[k] - 1 for k in ["H", "D", "A"]}
    best_bet = max(value, key=value.get)
    edge = value[best_bet]

    if edge > 0:
        full_kelly = edge / (odds_dict[best_bet] - 1)
        kelly_frac = min(full_kelly * 0.5, 0.5)
    else:
        kelly_frac = 0.0

    kelly_stake = st.session_state.bankroll * kelly_frac

    st.subheader("Value Bet")
    value_df = pd.DataFrame({
        "AI": [f"{probs['H']:.1%}", f"{probs['D']:.1%}", f"{probs['A']:.1%}"],
        "Odds": [f"{odds_home:.2f}", f"{odds_draw:.2f}", f"{odds_away:.2f}"],
        "Value": [f"{value['H']:.1%}", f"{value['D']:.1%}", f"{value['A']:.1%}"],
    }, index=["Hemma", "Oavgjort", "Borta"])
    st.dataframe(value_df.style.highlight_max(axis=0, subset="Value"), use_container_width=True)

    if edge > 0.05:
        st.success(f"VALUE BET: {best_bet.upper()} – edge {edge:.1%}")
    else:
        st.warning("Inget tydligt value bet.")

    st.subheader("Kelly (halv-Kelly)")
    kc1, kc2 = st.columns(2)
    with kc1:
        st.metric("Kelly-fraktion", f"{kelly_frac:.1%}")
    with kc2:
        st.metric("Rekommenderad insats", f"kr {kelly_stake:,.0f}")

    with st.expander("Bankrulle & Satsa", expanded=True):
        b1, b2 = st.columns(2)
        with b1:
            st.metric("Nuvarande bankrulle", f"kr {st.session_state.bankroll:,.0f}")
        with b2:
            stake = st.number_input("Satsa (valfritt)", min_value=0.0, value=float(kelly_stake), step=100.0, key="stake_input")
        if st.button("Satsa & Lägg till i historik", use_container_width=True):
            st.session_state.bets.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "match": f"{home_team} vs {away_team}",
                "bet": best_bet.upper(),
                "odds": odds_dict[best_bet],
                "stake": stake,
                "prob": probs[best_bet],
                "status": "Väntar",
            })
            st.success(f"Satsning på {best_bet.upper()} registrerad!")

    if st.session_state.bets:
        st.subheader("Satsningshistorik")
        bets_df = pd.DataFrame(st.session_state.bets)
        st.dataframe(bets_df, use_container_width=True)

        total_stake = sum(b["stake"] for b in st.session_state.bets if b.get("status") == "Väntar")
        st.metric("Totalt satsat (väntar)", f"kr {total_stake:,.0f}")

        if st.button("Exportera historik till CSV"):
            bets_df.to_csv("bets_history.csv", index=False, encoding="utf-8")
            st.success("Sparad som bets_history.csv i projektroten.")

    with st.expander("Målmatris"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=range(matrix.shape[1]), yticklabels=range(matrix.shape[0]), ax=ax)
        ax.set_xlabel("Bortamål")
        ax.set_ylabel("Hemmål")
        st.pyplot(fig)

# ======================
# 🔧 API-yta / Debugpanel
# ======================
with st.sidebar.expander("⚙️ Systemstatus & Debug", expanded=False):
    # 2.1 Vald bookmaker (sparas i session_state)
    if "bookmaker" not in st.session_state:
        st.session_state.bookmaker = "bet365"
    st.session_state.bookmaker = st.selectbox(
        "Bookmaker (visning/preferens)",
        ["bet365", "pinnacle", "williamhill", "unibet", "betfair"],
        index=["bet365", "pinnacle", "williamhill", "unibet", "betfair"].index(st.session_state.bookmaker),
        help="Påverkar inte API-kallet om du inte använder det i din parse-funktion, men lagras för UI/logik."
    )

    # 2.2 Senaste odds-hämtning (hantera float epoch eller datetime)
    last_ts = st.session_state.get("last_fetch_ts")
    if last_ts:
        try:
            if isinstance(last_ts, (int, float)):
                last_dt_utc = datetime.utcfromtimestamp(last_ts)
                secs_ago = int((datetime.utcnow() - last_dt_utc).total_seconds())
                st.metric("Senast hämtat (UTC)", last_dt_utc.strftime("%Y-%m-%d %H:%M:%S"), f"{secs_ago}s sedan")
            elif isinstance(last_ts, datetime):
                secs_ago = int((datetime.utcnow() - last_ts).total_seconds())
                st.metric("Senast hämtat (UTC)", last_ts.strftime("%Y-%m-%d %H:%M:%S"), f"{secs_ago}s sedan")
            else:
                st.write(f"Senast hämtat: {last_ts}")
        except Exception:
            st.write(f"Senast hämtat: {last_ts}")
    else:
        st.caption("Inga odds hämtade ännu i denna session.")

    # 2.3 Odds API kvota (om hämta_odds sparade headers)
    odds_headers = st.session_state.get("odds_headers", {})
    if odds_headers:
        st.write("**Odds API kvot**")
        st.json(odds_headers)

    # 2.4 Metoder i Bet365API
    st.write("**Bet365API – metoder**")
    try:
        api_methods = [m for m in dir(Bet365API) if callable(getattr(Bet365API, m)) and not m.startswith("_")]
        st.write(", ".join(api_methods) if api_methods else "Inga publika metoder hittade.")
    except Exception as e:
        st.write(f"Kunde inte läsa metoder: {e}")

    # 2.5 Laddade modeller/parametrar på disk
    try:
        model_files = sorted([p.name for p in MODELL_DIR.glob("*.pkl")])
        param_files = sorted([p.name for p in PARAM_DIR.glob("*_parametrar.json")])
        st.write("**Modellfiler**")
        st.write(model_files if model_files else "Inga .pkl ännu.")
        st.write("**Parametrar**")
        st.write(param_files if param_files else "Inga *_parametrar.json ännu.")
    except Exception as e:
        st.write(f"Kunde inte läsa modell/param-kataloger: {e}")

    # 2.6 Session state-snapshot (nyckel, typ, förhandsvisning)
    st.write("**Session state**")
    ss_rows = []
    for k, v in st.session_state.items():
        # begränsa förhandsvisning
        try:
            preview = repr(v)
            if len(preview) > 120:
                preview = preview[:117] + "..."
        except Exception:
            preview = "<unrepr>"
        ss_rows.append({
            "key": k,
            "type": type(v).__name__,
            "preview": preview
        })
    if ss_rows:
        ss_df = pd.DataFrame(ss_rows).sort_values("key")
        st.dataframe(ss_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Session state är tom.")

st.caption("Kelly Betting AI © 2025 - Spela ansvarsfullt")
