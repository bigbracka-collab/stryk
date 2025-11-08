# src/stryktips/ui.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import List, Dict
from .models import StryktipsMatch
from .generator import build_system
from .reducer import filter_rows_by_upsets

def stryktips_panel(matches: List[StryktipsMatch], default_budget: float = 512.0, row_cost: float = 1.0):
    st.header("🧩 Stryktips – systemgenerator")

    # Visa data och låt användaren finjustera budget/parametrar
    df = pd.DataFrame([{
        "Nr": m.idx, "Match": f"{m.home} - {m.away}",
        "Odds 1": m.odds_1, "Odds X": m.odds_X, "Odds 2": m.odds_2,
        "P(1)": m.p1, "P(X)": m.pX, "P(2)": m.p2
    } for m in matches])

    st.dataframe(df, use_container_width=True)

    with st.expander("Parametrar"):
        budget = st.number_input("Budget (kr)", min_value=row_cost, step=row_cost, value=float(default_budget))
        min_edge = st.slider("Min edge för spik", 0.0, 0.2, 0.02, 0.01)
        prefer_value = st.checkbox("Prioritera value över ren sannolikhet", value=True)
        max_width = st.select_slider("Max gardering per match", options=[1,2,3], value=3)
        upset_min = st.number_input("Reduceringsregel: min skrällar", 0, 13, 0)
        upset_max = st.number_input("Reduceringsregel: max skrällar", 0, 13, 13)

    if st.button("Generera system", use_container_width=True):
        sys = build_system(
            matches=matches, budget=budget, row_cost=row_cost,
            prefer_value=prefer_value, min_edge=min_edge, max_width=max_width
        )
        st.success(f"System klart: {sys.rows()} rader, kostnad {sys.cost():.0f} kr")

        # Visa sammanfattning per match
        summary = []
        for m in matches:
            outs = "".join(sys.choices[m.idx].outcomes)
            summary.append({"Nr": m.idx, "Match": f"{m.home}-{m.away}", "Val": outs})
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        rows = sys.to_rows()
        # Reducera
        reduced = filter_rows_by_upsets(rows, upset_min=upset_min, upset_max=upset_max)
        st.info(f"Reducerat: {len(reduced)} rader kvar (från {len(rows)})")

        # Export
        if st.button("Exportera kupongrader (CSV)", use_container_width=True):
            out_df = pd.DataFrame({"rad": reduced})
            out_df.to_csv("stryktips_rows.csv", index=False, encoding="utf-8")
            st.success("Sparat: stryktips_rows.csv")
